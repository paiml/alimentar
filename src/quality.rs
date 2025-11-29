//! Data quality assessment for ML pipelines
//!
//! Detects data quality issues including missing values, outliers,
//! duplicates, and schema problems.
//!
//! # 100-Point Quality Scoring System (GH-6)
//!
//! Based on the Toyota Way principles of Jidoka (built-in quality) and
//! the Doctest Corpus QA Checklist for Publication.
//!
//! ## Severity Weights
//! - **Critical (2.0x)**: Blocks publication - data integrity failures
//! - **High (1.5x)**: Major issues requiring immediate attention
//! - **Medium (1.0x)**: Standard issues to address before publication
//! - **Low (0.5x)**: Minor issues, informational
//!
//! ## Letter Grades
//! - **A (95-100)**: Publish immediately
//! - **B (85-94)**: Publish with documented caveats
//! - **C (70-84)**: Remediation required before publication
//! - **D (50-69)**: Major rework needed
//! - **F (<50)**: Do not publish
//!
//! # Example
//!
//! ```ignore
//! use alimentar::quality::{QualityChecker, QualityScore};
//!
//! let checker = QualityChecker::new()
//!     .max_null_ratio(0.1)
//!     .max_duplicate_ratio(0.05);
//!
//! let report = checker.check(&dataset)?;
//! let score = QualityScore::from_report(&report);
//! println!("Grade: {} ({})", score.grade, score.score);
//! ```
//!
//! # References
//! - [1] Batini & Scannapieco (2016). Data and Information Quality.
//! - [6] Hynes et al. (2017). The Data Linter. NIPS Workshop on ML Systems.

// Statistical computation and internal methods
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::unused_self)]
#![allow(clippy::if_not_else)]

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::Result,
};

// ═══════════════════════════════════════════════════════════════════════════════
// 100-Point Quality Scoring System (GH-6)
// ═══════════════════════════════════════════════════════════════════════════════

/// Severity levels for quality issues per QA checklist
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Critical issues block publication (2.0x weight)
    Critical,
    /// High priority issues (1.5x weight)
    High,
    /// Medium priority issues (1.0x weight)
    Medium,
    /// Low priority issues (0.5x weight)
    Low,
}

impl Severity {
    /// Get the weight multiplier for this severity
    #[must_use]
    pub fn weight(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }

    /// Get the base point value for this severity
    #[must_use]
    pub fn base_points(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Medium => write!(f, "Medium"),
            Self::Low => write!(f, "Low"),
        }
    }
}

/// Letter grades for dataset quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LetterGrade {
    /// A (95-100): Publish immediately
    A,
    /// B (85-94): Publish with documented caveats
    B,
    /// C (70-84): Remediation required before publication
    C,
    /// D (50-69): Major rework needed
    D,
    /// F (<50): Do not publish
    F,
}

impl LetterGrade {
    /// Create a letter grade from a numeric score (0-100)
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 95.0 => Self::A,
            s if s >= 85.0 => Self::B,
            s if s >= 70.0 => Self::C,
            s if s >= 50.0 => Self::D,
            _ => Self::F,
        }
    }

    /// Get the publication decision for this grade
    #[must_use]
    pub fn publication_decision(&self) -> &'static str {
        match self {
            Self::A => "Publish immediately",
            Self::B => "Publish with documented caveats",
            Self::C => "Remediation required before publication",
            Self::D => "Major rework needed",
            Self::F => "Do not publish",
        }
    }

    /// Check if this grade allows publication
    #[must_use]
    pub fn is_publishable(&self) -> bool {
        matches!(self, Self::A | Self::B)
    }
}

impl fmt::Display for LetterGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::F => write!(f, "F"),
        }
    }
}

/// A scored quality check item from the 100-point checklist
#[derive(Debug, Clone)]
pub struct ChecklistItem {
    /// Unique identifier (e.g., "1", "25", "53")
    pub id: u8,
    /// Check description
    pub description: String,
    /// Pass/fail status
    pub passed: bool,
    /// Severity level
    pub severity: Severity,
    /// Suggestion for improvement if failed
    pub suggestion: Option<String>,
}

impl ChecklistItem {
    /// Create a new checklist item
    #[must_use]
    pub fn new(id: u8, description: impl Into<String>, severity: Severity, passed: bool) -> Self {
        Self {
            id,
            description: description.into(),
            passed,
            severity,
            suggestion: None,
        }
    }

    /// Add a suggestion for improvement
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Get the points earned (0 if failed, severity points if passed)
    #[must_use]
    pub fn points_earned(&self) -> f64 {
        if self.passed {
            self.severity.base_points()
        } else {
            0.0
        }
    }

    /// Get the maximum possible points for this item
    #[must_use]
    pub fn max_points(&self) -> f64 {
        self.severity.base_points()
    }
}

/// Complete quality score with breakdown
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Numeric score (0-100)
    pub score: f64,
    /// Letter grade
    pub grade: LetterGrade,
    /// Total points earned
    pub points_earned: f64,
    /// Maximum possible points
    pub max_points: f64,
    /// Individual checklist items
    pub checklist: Vec<ChecklistItem>,
    /// Summary statistics by severity
    pub severity_breakdown: HashMap<Severity, SeverityStats>,
}

/// Statistics for a severity level
#[derive(Debug, Clone, Default)]
pub struct SeverityStats {
    /// Number of checks at this severity
    pub total: usize,
    /// Number of passed checks
    pub passed: usize,
    /// Number of failed checks
    pub failed: usize,
    /// Points earned at this severity
    pub points_earned: f64,
    /// Maximum possible points at this severity
    pub max_points: f64,
}

impl QualityScore {
    /// Create a quality score from checklist items
    #[must_use]
    pub fn from_checklist(checklist: Vec<ChecklistItem>) -> Self {
        let mut severity_breakdown: HashMap<Severity, SeverityStats> = HashMap::new();

        let mut points_earned = 0.0;
        let mut max_points = 0.0;

        for item in &checklist {
            let stats = severity_breakdown.entry(item.severity).or_default();

            stats.total += 1;
            stats.max_points += item.max_points();

            if item.passed {
                stats.passed += 1;
                stats.points_earned += item.points_earned();
                points_earned += item.points_earned();
            } else {
                stats.failed += 1;
            }

            max_points += item.max_points();
        }

        let score = if max_points > 0.0 {
            (points_earned / max_points * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        };

        let grade = LetterGrade::from_score(score);

        Self {
            score,
            grade,
            points_earned,
            max_points,
            checklist,
            severity_breakdown,
        }
    }

    /// Get failed items for actionable suggestions
    #[must_use]
    pub fn failed_items(&self) -> Vec<&ChecklistItem> {
        self.checklist.iter().filter(|item| !item.passed).collect()
    }

    /// Get critical failures (blocks publication)
    #[must_use]
    pub fn critical_failures(&self) -> Vec<&ChecklistItem> {
        self.checklist
            .iter()
            .filter(|item| !item.passed && item.severity == Severity::Critical)
            .collect()
    }

    /// Check if there are any critical failures
    #[must_use]
    pub fn has_critical_failures(&self) -> bool {
        self.checklist
            .iter()
            .any(|item| !item.passed && item.severity == Severity::Critical)
    }

    /// Generate a badge URL for shields.io
    #[must_use]
    pub fn badge_url(&self) -> String {
        let color = match self.grade {
            LetterGrade::A => "brightgreen",
            LetterGrade::B => "green",
            LetterGrade::C => "yellow",
            LetterGrade::D => "orange",
            LetterGrade::F => "red",
        };
        format!(
            "https://img.shields.io/badge/data_quality-{}_({:.0}%25)-{}",
            self.grade, self.score, color
        )
    }

    /// Generate JSON output for CI/CD integration
    #[must_use]
    pub fn to_json(&self) -> String {
        let failed_items: Vec<_> = self
            .failed_items()
            .iter()
            .map(|item| {
                format!(
                    r#"    {{"id": {}, "description": "{}", "severity": "{}", "suggestion": {}}}"#,
                    item.id,
                    item.description.replace('"', "\\\""),
                    item.severity,
                    item.suggestion
                        .as_ref()
                        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                        .unwrap_or_else(|| "null".to_string())
                )
            })
            .collect();

        format!(
            r#"{{
  "score": {:.2},
  "grade": "{}",
  "is_publishable": {},
  "decision": "{}",
  "points_earned": {:.2},
  "max_points": {:.2},
  "critical_failures": {},
  "failed_items": [
{}
  ],
  "badge_url": "{}"
}}"#,
            self.score,
            self.grade,
            self.grade.is_publishable(),
            self.grade.publication_decision(),
            self.points_earned,
            self.max_points,
            self.has_critical_failures(),
            failed_items.join(",\n"),
            self.badge_url()
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quality Profiles (GH-10)
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality profile for customizing scoring rules per data type.
///
/// Different data types (doctest corpora, ML training sets, time series, etc.)
/// have different expectations. For example:
/// - Doctest corpus: `source` and `version` columns are expected to be constant
/// - ML training: features should have high variance, labels can be categorical
/// - Time series: timestamps should be unique and sequential
///
/// # Example
///
/// ```ignore
/// let profile = QualityProfile::doctest_corpus();
/// let score = profile.score_report(&report);
/// ```
#[derive(Debug, Clone)]
pub struct QualityProfile {
    /// Profile name for display
    pub name: String,
    /// Description of what this profile is for
    pub description: String,
    /// Columns that are expected to be constant (not penalized)
    pub expected_constant_columns: HashSet<String>,
    /// Columns where high null ratio is acceptable
    pub nullable_columns: HashSet<String>,
    /// Maximum acceptable null ratio (default: 0.1)
    pub max_null_ratio: f64,
    /// Maximum acceptable duplicate ratio (default: 0.5)
    pub max_duplicate_ratio: f64,
    /// Minimum cardinality before flagging as low (default: 2)
    pub min_cardinality: usize,
    /// Maximum outlier ratio to report (default: 0.05)
    pub max_outlier_ratio: f64,
    /// Maximum duplicate row ratio (default: 0.01)
    pub max_duplicate_row_ratio: f64,
    /// Whether to penalize constant columns not in expected list
    pub penalize_unexpected_constants: bool,
    /// Whether this profile requires a signature column (for doctest)
    pub require_signature: bool,
}

impl Default for QualityProfile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: "General-purpose quality profile".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.1,
            max_duplicate_ratio: 0.5,
            min_cardinality: 2,
            max_outlier_ratio: 0.05,
            max_duplicate_row_ratio: 0.01,
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }
}

impl QualityProfile {
    /// Create a new profile with custom name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Get profile by name
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "default" => Some(Self::default()),
            "doctest-corpus" | "doctest" => Some(Self::doctest_corpus()),
            "ml-training" | "ml" => Some(Self::ml_training()),
            "time-series" | "timeseries" => Some(Self::time_series()),
            _ => None,
        }
    }

    /// List available profile names
    #[must_use]
    pub fn available_profiles() -> Vec<&'static str> {
        vec!["default", "doctest-corpus", "ml-training", "time-series"]
    }

    /// Doctest corpus profile - for Python doctest extraction datasets.
    ///
    /// Expects:
    /// - `source` and `version` columns to be constant (single crate/version)
    /// - `signature` column may have nulls (module-level doctests)
    /// - `input`, `expected`, `function` should be non-null
    #[must_use]
    pub fn doctest_corpus() -> Self {
        let mut expected_constants = HashSet::new();
        expected_constants.insert("source".to_string());
        expected_constants.insert("version".to_string());

        let mut nullable = HashSet::new();
        nullable.insert("signature".to_string()); // Module-level doctests have no signature

        Self {
            name: "doctest-corpus".to_string(),
            description: "Profile for Python doctest extraction datasets".to_string(),
            expected_constant_columns: expected_constants,
            nullable_columns: nullable,
            max_null_ratio: 0.05, // Stricter for doctest data
            max_duplicate_ratio: 0.3, // Some duplicate inputs are normal
            min_cardinality: 2,
            max_outlier_ratio: 0.05,
            max_duplicate_row_ratio: 0.0, // No exact duplicate rows allowed
            penalize_unexpected_constants: true,
            require_signature: false, // Relaxed - signature nulls are OK for module doctests
        }
    }

    /// ML training profile - for machine learning datasets.
    ///
    /// Expects:
    /// - Features to have reasonable variance
    /// - Labels can be categorical (low cardinality OK)
    /// - No null values in features or labels
    #[must_use]
    pub fn ml_training() -> Self {
        Self {
            name: "ml-training".to_string(),
            description: "Profile for machine learning training datasets".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.0, // No nulls allowed in training data
            max_duplicate_ratio: 0.8, // Higher tolerance for categorical features
            min_cardinality: 2,
            max_outlier_ratio: 0.1, // More tolerant of outliers
            max_duplicate_row_ratio: 0.01,
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }

    /// Time series profile - for temporal data.
    ///
    /// Expects:
    /// - Timestamp column should be unique
    /// - Data should have temporal patterns
    #[must_use]
    pub fn time_series() -> Self {
        Self {
            name: "time-series".to_string(),
            description: "Profile for time series datasets".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.05,
            max_duplicate_ratio: 0.5,
            min_cardinality: 2,
            max_outlier_ratio: 0.1, // Time series often have outliers
            max_duplicate_row_ratio: 0.0, // No duplicate rows (each timestamp unique)
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add an expected constant column
    #[must_use]
    pub fn with_expected_constant(mut self, column: impl Into<String>) -> Self {
        self.expected_constant_columns.insert(column.into());
        self
    }

    /// Add a nullable column
    #[must_use]
    pub fn with_nullable(mut self, column: impl Into<String>) -> Self {
        self.nullable_columns.insert(column.into());
        self
    }

    /// Set max null ratio
    #[must_use]
    pub fn with_max_null_ratio(mut self, ratio: f64) -> Self {
        self.max_null_ratio = ratio;
        self
    }

    /// Set max duplicate ratio
    #[must_use]
    pub fn with_max_duplicate_ratio(mut self, ratio: f64) -> Self {
        self.max_duplicate_ratio = ratio;
        self
    }

    /// Check if a column is expected to be constant
    #[must_use]
    pub fn is_expected_constant(&self, column: &str) -> bool {
        self.expected_constant_columns.contains(column)
    }

    /// Check if a column is allowed to have nulls
    #[must_use]
    pub fn is_nullable(&self, column: &str) -> bool {
        self.nullable_columns.contains(column)
    }

    /// Get effective null threshold for a column
    #[must_use]
    pub fn null_threshold_for(&self, column: &str) -> f64 {
        if self.is_nullable(column) {
            1.0 // Allow up to 100% nulls for nullable columns
        } else {
            self.max_null_ratio
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quality Issues
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of data quality issues
#[derive(Debug, Clone, PartialEq)]
pub enum QualityIssue {
    /// Column has high percentage of null/missing values
    HighNullRatio {
        /// Column name
        column: String,
        /// Actual null ratio
        null_ratio: f64,
        /// Configured threshold
        threshold: f64,
    },
    /// Column has high percentage of duplicate values
    HighDuplicateRatio {
        /// Column name
        column: String,
        /// Actual duplicate ratio
        duplicate_ratio: f64,
        /// Configured threshold
        threshold: f64,
    },
    /// Column has very low cardinality (potential constant)
    LowCardinality {
        /// Column name
        column: String,
        /// Number of unique values
        unique_count: usize,
        /// Total row count
        total_count: usize,
    },
    /// Column has potential outliers (IQR method)
    OutliersDetected {
        /// Column name
        column: String,
        /// Number of outliers
        outlier_count: usize,
        /// Ratio of outliers
        outlier_ratio: f64,
    },
    /// Dataset has duplicate rows
    DuplicateRows {
        /// Number of duplicate rows
        duplicate_count: usize,
        /// Ratio of duplicate rows
        duplicate_ratio: f64,
    },
    /// Column has constant value (zero variance)
    ConstantColumn {
        /// Column name
        column: String,
        /// The constant value
        value: String,
    },
    /// Schema has no columns
    EmptySchema,
    /// Dataset is empty
    EmptyDataset,
}

impl QualityIssue {
    /// Get severity level (1-5, higher is worse)
    pub fn severity(&self) -> u8 {
        match self {
            Self::EmptySchema | Self::EmptyDataset => 5,
            Self::ConstantColumn { .. } => 4,
            Self::HighNullRatio { null_ratio, .. } if *null_ratio > 0.5 => 4,
            Self::HighNullRatio { .. } => 3,
            Self::OutliersDetected { outlier_ratio, .. } if *outlier_ratio > 0.1 => 3,
            Self::OutliersDetected { .. }
            | Self::HighDuplicateRatio { .. }
            | Self::DuplicateRows { .. } => 2,
            Self::LowCardinality { .. } => 1,
        }
    }

    /// Get column name if applicable
    pub fn column(&self) -> Option<&str> {
        match self {
            Self::HighNullRatio { column, .. }
            | Self::HighDuplicateRatio { column, .. }
            | Self::LowCardinality { column, .. }
            | Self::OutliersDetected { column, .. }
            | Self::ConstantColumn { column, .. } => Some(column),
            _ => None,
        }
    }
}

/// Quality statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnQuality {
    /// Column name
    pub name: String,
    /// Total row count
    pub total_count: usize,
    /// Null/missing count
    pub null_count: usize,
    /// Null ratio (0-1)
    pub null_ratio: f64,
    /// Number of unique values
    pub unique_count: usize,
    /// Unique ratio (unique/total)
    pub unique_ratio: f64,
    /// Number of duplicate values (non-unique occurrences)
    pub duplicate_count: usize,
    /// Duplicate ratio
    pub duplicate_ratio: f64,
    /// Number of outliers (for numeric columns)
    pub outlier_count: Option<usize>,
    /// Basic stats for numeric columns
    pub numeric_stats: Option<NumericStats>,
}

impl ColumnQuality {
    /// Check if column is constant (single unique value)
    pub fn is_constant(&self) -> bool {
        self.unique_count <= 1 && self.total_count > 0
    }

    /// Check if column is mostly null
    pub fn is_mostly_null(&self, threshold: f64) -> bool {
        self.null_ratio >= threshold
    }
}

/// Basic statistics for numeric columns
#[derive(Debug, Clone)]
pub struct NumericStats {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 25th percentile (Q1)
    pub q1: f64,
    /// 50th percentile (median)
    pub median: f64,
    /// 75th percentile (Q3)
    pub q3: f64,
}

impl NumericStats {
    /// Calculate IQR (Interquartile Range)
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Get lower bound for outliers (Q1 - 1.5*IQR)
    pub fn outlier_lower_bound(&self) -> f64 {
        self.q1 - 1.5 * self.iqr()
    }

    /// Get upper bound for outliers (Q3 + 1.5*IQR)
    pub fn outlier_upper_bound(&self) -> f64 {
        self.q3 + 1.5 * self.iqr()
    }
}

/// Overall data quality report
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Total row count
    pub row_count: usize,
    /// Total column count
    pub column_count: usize,
    /// Per-column quality statistics
    pub columns: HashMap<String, ColumnQuality>,
    /// Detected issues
    pub issues: Vec<QualityIssue>,
    /// Overall quality score (0-100)
    pub score: f64,
    /// Number of duplicate rows
    pub duplicate_row_count: usize,
}

impl QualityReport {
    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }

    /// Get issues for a specific column
    pub fn column_issues(&self, column: &str) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|i| i.column() == Some(column))
            .collect()
    }

    /// Get maximum severity among all issues
    pub fn max_severity(&self) -> u8 {
        self.issues.iter().map(|i| i.severity()).max().unwrap_or(0)
    }

    /// Get columns with issues
    pub fn problematic_columns(&self) -> Vec<&str> {
        self.issues
            .iter()
            .filter_map(|i| i.column())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Configuration thresholds for quality checking
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Maximum acceptable null ratio (default: 0.1)
    pub max_null_ratio: f64,
    /// Maximum acceptable duplicate ratio (default: 0.5)
    pub max_duplicate_ratio: f64,
    /// Minimum cardinality to not flag as low (default: 2)
    pub min_cardinality: usize,
    /// Maximum outlier ratio to report (default: 0.05)
    pub max_outlier_ratio: f64,
    /// Maximum duplicate row ratio (default: 0.01)
    pub max_duplicate_row_ratio: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_null_ratio: 0.1,
            max_duplicate_ratio: 0.5,
            min_cardinality: 2,
            max_outlier_ratio: 0.05,
            max_duplicate_row_ratio: 0.01,
        }
    }
}

/// Data quality checker
pub struct QualityChecker {
    thresholds: QualityThresholds,
    check_outliers: bool,
    check_duplicates: bool,
}

impl Default for QualityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityChecker {
    /// Create a new quality checker with default thresholds
    pub fn new() -> Self {
        Self {
            thresholds: QualityThresholds::default(),
            check_outliers: true,
            check_duplicates: true,
        }
    }

    /// Set maximum null ratio threshold
    #[must_use]
    pub fn max_null_ratio(mut self, ratio: f64) -> Self {
        self.thresholds.max_null_ratio = ratio;
        self
    }

    /// Set maximum duplicate ratio threshold
    #[must_use]
    pub fn max_duplicate_ratio(mut self, ratio: f64) -> Self {
        self.thresholds.max_duplicate_ratio = ratio;
        self
    }

    /// Set minimum cardinality threshold
    #[must_use]
    pub fn min_cardinality(mut self, min: usize) -> Self {
        self.thresholds.min_cardinality = min;
        self
    }

    /// Set maximum outlier ratio threshold
    #[must_use]
    pub fn max_outlier_ratio(mut self, ratio: f64) -> Self {
        self.thresholds.max_outlier_ratio = ratio;
        self
    }

    /// Enable/disable outlier checking
    #[must_use]
    pub fn with_outlier_check(mut self, enabled: bool) -> Self {
        self.check_outliers = enabled;
        self
    }

    /// Enable/disable duplicate row checking
    #[must_use]
    pub fn with_duplicate_check(mut self, enabled: bool) -> Self {
        self.check_duplicates = enabled;
        self
    }

    /// Check dataset quality
    pub fn check(&self, dataset: &ArrowDataset) -> Result<QualityReport> {
        let schema = dataset.schema();
        let mut issues = Vec::new();

        // Check for empty schema
        if schema.fields().is_empty() {
            issues.push(QualityIssue::EmptySchema);
            return Ok(QualityReport {
                row_count: 0,
                column_count: 0,
                columns: HashMap::new(),
                issues,
                score: 0.0,
                duplicate_row_count: 0,
            });
        }

        // Collect all data
        let (column_data, row_count) = self.collect_data(dataset);

        // Check for empty dataset
        if row_count == 0 {
            issues.push(QualityIssue::EmptyDataset);
            return Ok(QualityReport {
                row_count: 0,
                column_count: schema.fields().len(),
                columns: HashMap::new(),
                issues,
                score: 0.0,
                duplicate_row_count: 0,
            });
        }

        // Analyze each column
        let mut columns = HashMap::new();
        for (col_name, values) in &column_data {
            let quality = self.analyze_column(col_name, values, row_count);

            // Check for issues
            if quality.null_ratio > self.thresholds.max_null_ratio {
                issues.push(QualityIssue::HighNullRatio {
                    column: col_name.clone(),
                    null_ratio: quality.null_ratio,
                    threshold: self.thresholds.max_null_ratio,
                });
            }

            if quality.duplicate_ratio > self.thresholds.max_duplicate_ratio {
                issues.push(QualityIssue::HighDuplicateRatio {
                    column: col_name.clone(),
                    duplicate_ratio: quality.duplicate_ratio,
                    threshold: self.thresholds.max_duplicate_ratio,
                });
            }

            if quality.unique_count < self.thresholds.min_cardinality && row_count > 1 {
                issues.push(QualityIssue::LowCardinality {
                    column: col_name.clone(),
                    unique_count: quality.unique_count,
                    total_count: row_count,
                });
            }

            if quality.is_constant() {
                let value = values
                    .iter()
                    .find(|v| v.is_some())
                    .map(|v| v.clone().unwrap_or_default())
                    .unwrap_or_default();
                issues.push(QualityIssue::ConstantColumn {
                    column: col_name.clone(),
                    value,
                });
            }

            if let Some(outlier_count) = quality.outlier_count {
                let outlier_ratio = outlier_count as f64 / row_count as f64;
                if outlier_ratio > self.thresholds.max_outlier_ratio {
                    issues.push(QualityIssue::OutliersDetected {
                        column: col_name.clone(),
                        outlier_count,
                        outlier_ratio,
                    });
                }
            }

            columns.insert(col_name.clone(), quality);
        }

        // Check for duplicate rows
        let duplicate_row_count = if self.check_duplicates {
            self.count_duplicate_rows(&column_data, row_count)
        } else {
            0
        };

        let duplicate_row_ratio = duplicate_row_count as f64 / row_count as f64;
        if duplicate_row_ratio > self.thresholds.max_duplicate_row_ratio {
            issues.push(QualityIssue::DuplicateRows {
                duplicate_count: duplicate_row_count,
                duplicate_ratio: duplicate_row_ratio,
            });
        }

        // Calculate quality score
        let score = self.calculate_score(&columns, &issues, row_count);

        Ok(QualityReport {
            row_count,
            column_count: schema.fields().len(),
            columns,
            issues,
            score,
            duplicate_row_count,
        })
    }

    /// Collect data from dataset as strings for analysis
    fn collect_data(
        &self,
        dataset: &ArrowDataset,
    ) -> (HashMap<String, Vec<Option<String>>>, usize) {
        use arrow::array::{Array, Float64Array, Int32Array, Int64Array, StringArray};

        let schema = dataset.schema();
        let mut data: HashMap<String, Vec<Option<String>>> = HashMap::new();
        let mut row_count = 0;

        for field in schema.fields() {
            data.insert(field.name().clone(), Vec::new());
        }

        for batch in dataset.iter() {
            row_count += batch.num_rows();

            for (col_idx, field) in schema.fields().iter().enumerate() {
                if let Some(col_data) = data.get_mut(field.name()) {
                    let array = batch.column(col_idx);

                    for i in 0..array.len() {
                        if array.is_null(i) {
                            col_data.push(None);
                        } else if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                            col_data.push(Some(arr.value(i).to_string()));
                        } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                            col_data.push(Some(arr.value(i).to_string()));
                        } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                            col_data.push(Some(arr.value(i).to_string()));
                        } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                            col_data.push(Some(arr.value(i).to_string()));
                        } else {
                            col_data.push(Some("?".to_string()));
                        }
                    }
                }
            }
        }

        (data, row_count)
    }

    /// Analyze a single column
    fn analyze_column(
        &self,
        name: &str,
        values: &[Option<String>],
        total_count: usize,
    ) -> ColumnQuality {
        let null_count = values.iter().filter(|v| v.is_none()).count();
        let null_ratio = if total_count > 0 {
            null_count as f64 / total_count as f64
        } else {
            0.0
        };

        // Count unique values
        let non_null_values: Vec<&str> = values.iter().filter_map(|v| v.as_deref()).collect();
        let unique_set: HashSet<&str> = non_null_values.iter().copied().collect();
        let unique_count = unique_set.len();
        let unique_ratio = if !non_null_values.is_empty() {
            unique_count as f64 / non_null_values.len() as f64
        } else {
            0.0
        };

        // Calculate duplicates
        let duplicate_count = non_null_values.len().saturating_sub(unique_count);
        let duplicate_ratio = if !non_null_values.is_empty() {
            duplicate_count as f64 / non_null_values.len() as f64
        } else {
            0.0
        };

        // Try to parse as numeric for outlier detection
        let (outlier_count, numeric_stats) = if self.check_outliers {
            self.analyze_numeric(&non_null_values)
        } else {
            (None, None)
        };

        ColumnQuality {
            name: name.to_string(),
            total_count,
            null_count,
            null_ratio,
            unique_count,
            unique_ratio,
            duplicate_count,
            duplicate_ratio,
            outlier_count,
            numeric_stats,
        }
    }

    /// Analyze numeric column for outliers and stats
    fn analyze_numeric(&self, values: &[&str]) -> (Option<usize>, Option<NumericStats>) {
        let numeric_values: Vec<f64> = values
            .iter()
            .filter_map(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .collect();

        if numeric_values.len() < 4 {
            return (None, None);
        }

        let mut sorted = numeric_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let mean = numeric_values.iter().sum::<f64>() / n as f64;

        let variance = numeric_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        let q1 = sorted[n / 4];
        let median = sorted[n / 2];
        let q3 = sorted[3 * n / 4];

        let stats = NumericStats {
            min,
            max,
            mean,
            std_dev,
            q1,
            median,
            q3,
        };

        // Count outliers using IQR method
        let lower = stats.outlier_lower_bound();
        let upper = stats.outlier_upper_bound();
        let outlier_count = numeric_values
            .iter()
            .filter(|&&v| v < lower || v > upper)
            .count();

        (Some(outlier_count), Some(stats))
    }

    /// Count duplicate rows
    fn count_duplicate_rows(
        &self,
        data: &HashMap<String, Vec<Option<String>>>,
        row_count: usize,
    ) -> usize {
        if data.is_empty() || row_count == 0 {
            return 0;
        }

        // Build row hashes
        let mut row_set: HashSet<String> = HashSet::new();
        let mut duplicates = 0;

        let columns: Vec<&String> = data.keys().collect();

        for i in 0..row_count {
            let row_key: String = columns
                .iter()
                .map(|col| {
                    data.get(*col)
                        .and_then(|v| v.get(i))
                        .map(|v| v.clone().unwrap_or_else(|| "NULL".to_string()))
                        .unwrap_or_else(|| "NULL".to_string())
                })
                .collect::<Vec<_>>()
                .join("|");

            if !row_set.insert(row_key) {
                duplicates += 1;
            }
        }

        duplicates
    }

    /// Calculate quality score (0-100)
    fn calculate_score(
        &self,
        columns: &HashMap<String, ColumnQuality>,
        issues: &[QualityIssue],
        row_count: usize,
    ) -> f64 {
        if row_count == 0 || columns.is_empty() {
            return 0.0;
        }

        let mut score = 100.0;

        // Deduct for null ratios
        let avg_null_ratio: f64 =
            columns.values().map(|c| c.null_ratio).sum::<f64>() / columns.len() as f64;
        score -= avg_null_ratio * 30.0;

        // Deduct for issues
        for issue in issues {
            score -= match issue.severity() {
                5 => 25.0,
                4 => 15.0,
                3 => 10.0,
                2 => 5.0,
                1 => 2.0,
                _ => 0.0,
            };
        }

        score.clamp(0.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    // ========== QualityIssue tests ==========

    #[test]
    fn test_issue_severity() {
        assert_eq!(QualityIssue::EmptySchema.severity(), 5);
        assert_eq!(QualityIssue::EmptyDataset.severity(), 5);

        let constant = QualityIssue::ConstantColumn {
            column: "x".to_string(),
            value: "1".to_string(),
        };
        assert_eq!(constant.severity(), 4);

        let high_null = QualityIssue::HighNullRatio {
            column: "x".to_string(),
            null_ratio: 0.6,
            threshold: 0.1,
        };
        assert_eq!(high_null.severity(), 4);

        let low_null = QualityIssue::HighNullRatio {
            column: "x".to_string(),
            null_ratio: 0.3,
            threshold: 0.1,
        };
        assert_eq!(low_null.severity(), 3);
    }

    #[test]
    fn test_issue_column() {
        let issue = QualityIssue::HighNullRatio {
            column: "test".to_string(),
            null_ratio: 0.5,
            threshold: 0.1,
        };
        assert_eq!(issue.column(), Some("test"));

        assert_eq!(QualityIssue::EmptySchema.column(), None);
    }

    // ========== ColumnQuality tests ==========

    #[test]
    fn test_column_quality_is_constant() {
        let mut quality = ColumnQuality {
            name: "test".to_string(),
            total_count: 100,
            null_count: 0,
            null_ratio: 0.0,
            unique_count: 1,
            unique_ratio: 0.01,
            duplicate_count: 99,
            duplicate_ratio: 0.99,
            outlier_count: None,
            numeric_stats: None,
        };

        assert!(quality.is_constant());

        quality.unique_count = 5;
        assert!(!quality.is_constant());
    }

    #[test]
    fn test_column_quality_mostly_null() {
        let quality = ColumnQuality {
            name: "test".to_string(),
            total_count: 100,
            null_count: 80,
            null_ratio: 0.8,
            unique_count: 5,
            unique_ratio: 0.25,
            duplicate_count: 15,
            duplicate_ratio: 0.75,
            outlier_count: None,
            numeric_stats: None,
        };

        assert!(quality.is_mostly_null(0.5));
        assert!(!quality.is_mostly_null(0.9));
    }

    // ========== NumericStats tests ==========

    #[test]
    fn test_numeric_stats_iqr() {
        let stats = NumericStats {
            min: 0.0,
            max: 100.0,
            mean: 50.0,
            std_dev: 25.0,
            q1: 25.0,
            median: 50.0,
            q3: 75.0,
        };

        assert!((stats.iqr() - 50.0).abs() < 0.01);
        assert!((stats.outlier_lower_bound() - (-50.0)).abs() < 0.01);
        assert!((stats.outlier_upper_bound() - 150.0).abs() < 0.01);
    }

    // ========== QualityReport tests ==========

    #[test]
    fn test_report_has_issues() {
        let report = QualityReport {
            row_count: 100,
            column_count: 2,
            columns: HashMap::new(),
            issues: vec![],
            score: 100.0,
            duplicate_row_count: 0,
        };
        assert!(!report.has_issues());

        let report_with_issues = QualityReport {
            row_count: 100,
            column_count: 2,
            columns: HashMap::new(),
            issues: vec![QualityIssue::EmptySchema],
            score: 50.0,
            duplicate_row_count: 0,
        };
        assert!(report_with_issues.has_issues());
    }

    #[test]
    fn test_report_max_severity() {
        let report = QualityReport {
            row_count: 100,
            column_count: 2,
            columns: HashMap::new(),
            issues: vec![
                QualityIssue::LowCardinality {
                    column: "x".to_string(),
                    unique_count: 1,
                    total_count: 100,
                },
                QualityIssue::ConstantColumn {
                    column: "y".to_string(),
                    value: "1".to_string(),
                },
            ],
            score: 80.0,
            duplicate_row_count: 0,
        };

        assert_eq!(report.max_severity(), 4);
    }

    // ========== QualityChecker tests ==========

    fn make_dataset(col1: Vec<Option<&str>>, col2: Vec<Option<i32>>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Int32, true),
        ]));

        let names: Vec<Option<&str>> = col1;
        let values: Vec<Option<i32>> = col2;

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    fn make_float_dataset(values: Vec<Option<f64>>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(values))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    #[test]
    fn test_checker_new() {
        let checker = QualityChecker::new();
        assert!((checker.thresholds.max_null_ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_checker_builder() {
        let checker = QualityChecker::new()
            .max_null_ratio(0.2)
            .max_duplicate_ratio(0.3)
            .min_cardinality(5);

        assert!((checker.thresholds.max_null_ratio - 0.2).abs() < 0.01);
        assert!((checker.thresholds.max_duplicate_ratio - 0.3).abs() < 0.01);
        assert_eq!(checker.thresholds.min_cardinality, 5);
    }

    #[test]
    fn test_checker_clean_data() {
        let dataset = make_dataset(
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
            vec![Some(1), Some(2), Some(3), Some(4)],
        );

        let checker = QualityChecker::new();
        let report = checker.check(&dataset).expect("check");

        assert_eq!(report.row_count, 4);
        assert_eq!(report.column_count, 2);
        assert!(report.score > 80.0);
    }

    #[test]
    fn test_checker_detects_nulls() {
        let dataset = make_dataset(
            vec![Some("a"), None, None, None, None],
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        );

        let checker = QualityChecker::new().max_null_ratio(0.5);
        let report = checker.check(&dataset).expect("check");

        let null_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| matches!(i, QualityIssue::HighNullRatio { .. }))
            .collect();

        assert_eq!(null_issues.len(), 1);
    }

    #[test]
    fn test_checker_detects_constant() {
        let dataset = make_dataset(
            vec![Some("same"), Some("same"), Some("same"), Some("same")],
            vec![Some(1), Some(2), Some(3), Some(4)],
        );

        let checker = QualityChecker::new();
        let report = checker.check(&dataset).expect("check");

        let constant_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| matches!(i, QualityIssue::ConstantColumn { .. }))
            .collect();

        assert_eq!(constant_issues.len(), 1);
    }

    #[test]
    fn test_checker_detects_duplicates() {
        let dataset = make_dataset(
            vec![Some("a"), Some("a"), Some("a"), Some("b")],
            vec![Some(1), Some(1), Some(1), Some(2)],
        );

        let checker = QualityChecker::new().max_duplicate_ratio(0.01);
        let report = checker.check(&dataset).expect("check");

        // Should detect duplicate rows
        assert!(report.duplicate_row_count > 0);
    }

    #[test]
    fn test_checker_detects_outliers() {
        // Create dataset with clear outliers
        let mut values: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
        values.push(Some(10000.0)); // outlier
        values.push(Some(-10000.0)); // outlier

        let dataset = make_float_dataset(values);

        let checker = QualityChecker::new().max_outlier_ratio(0.01);
        let report = checker.check(&dataset).expect("check");

        let outlier_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| matches!(i, QualityIssue::OutliersDetected { .. }))
            .collect();

        assert!(!outlier_issues.is_empty());
    }

    #[test]
    fn test_checker_empty_dataset() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(Vec::<i32>::new()))],
        )
        .expect("batch");
        let dataset = ArrowDataset::from_batch(batch).expect("dataset");

        let checker = QualityChecker::new();
        let report = checker.check(&dataset).expect("check");

        assert!(report.issues.contains(&QualityIssue::EmptyDataset));
        assert_eq!(report.score, 0.0);
    }

    #[test]
    fn test_checker_score_decreases_with_issues() {
        let clean_dataset = make_dataset(
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
            vec![Some(1), Some(2), Some(3), Some(4)],
        );

        let dirty_dataset = make_dataset(
            vec![Some("same"), Some("same"), None, None],
            vec![Some(1), Some(1), None, None],
        );

        let checker = QualityChecker::new();
        let clean_report = checker.check(&clean_dataset).expect("check");
        let dirty_report = checker.check(&dirty_dataset).expect("check");

        assert!(clean_report.score > dirty_report.score);
    }

    #[test]
    fn test_checker_column_issues() {
        let dataset = make_dataset(
            vec![None, None, None, None],
            vec![Some(1), Some(2), Some(3), Some(4)],
        );

        let checker = QualityChecker::new();
        let report = checker.check(&dataset).expect("check");

        let name_issues = report.column_issues("name");
        assert!(!name_issues.is_empty());

        let value_issues = report.column_issues("value");
        // value column should have fewer issues
        assert!(value_issues.len() < name_issues.len());
    }

    #[test]
    fn test_checker_problematic_columns() {
        let dataset = make_dataset(
            vec![None, None, None, None],
            vec![Some(1), Some(1), Some(1), Some(1)],
        );

        let checker = QualityChecker::new();
        let report = checker.check(&dataset).expect("check");

        let problematic = report.problematic_columns();
        assert!(problematic.contains(&"name"));
        assert!(problematic.contains(&"value"));
    }

    #[test]
    fn test_checker_disable_outliers() {
        let mut values: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
        values.push(Some(10000.0));

        let dataset = make_float_dataset(values);

        let checker = QualityChecker::new()
            .with_outlier_check(false)
            .max_outlier_ratio(0.001);
        let report = checker.check(&dataset).expect("check");

        let outlier_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| matches!(i, QualityIssue::OutliersDetected { .. }))
            .collect();

        assert!(outlier_issues.is_empty());
    }

    // ========== 100-Point Quality Scoring System Tests (GH-6) ==========

    #[test]
    fn test_severity_weights() {
        assert!((Severity::Critical.weight() - 2.0).abs() < 0.01);
        assert!((Severity::High.weight() - 1.5).abs() < 0.01);
        assert!((Severity::Medium.weight() - 1.0).abs() < 0.01);
        assert!((Severity::Low.weight() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_severity_base_points() {
        assert!((Severity::Critical.base_points() - 2.0).abs() < 0.01);
        assert!((Severity::High.base_points() - 1.5).abs() < 0.01);
        assert!((Severity::Medium.base_points() - 1.0).abs() < 0.01);
        assert!((Severity::Low.base_points() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Critical), "Critical");
        assert_eq!(format!("{}", Severity::High), "High");
        assert_eq!(format!("{}", Severity::Medium), "Medium");
        assert_eq!(format!("{}", Severity::Low), "Low");
    }

    #[test]
    fn test_letter_grade_from_score() {
        assert_eq!(LetterGrade::from_score(100.0), LetterGrade::A);
        assert_eq!(LetterGrade::from_score(95.0), LetterGrade::A);
        assert_eq!(LetterGrade::from_score(94.9), LetterGrade::B);
        assert_eq!(LetterGrade::from_score(85.0), LetterGrade::B);
        assert_eq!(LetterGrade::from_score(84.9), LetterGrade::C);
        assert_eq!(LetterGrade::from_score(70.0), LetterGrade::C);
        assert_eq!(LetterGrade::from_score(69.9), LetterGrade::D);
        assert_eq!(LetterGrade::from_score(50.0), LetterGrade::D);
        assert_eq!(LetterGrade::from_score(49.9), LetterGrade::F);
        assert_eq!(LetterGrade::from_score(0.0), LetterGrade::F);
    }

    #[test]
    fn test_letter_grade_publication_decision() {
        assert_eq!(LetterGrade::A.publication_decision(), "Publish immediately");
        assert_eq!(
            LetterGrade::B.publication_decision(),
            "Publish with documented caveats"
        );
        assert_eq!(
            LetterGrade::C.publication_decision(),
            "Remediation required before publication"
        );
        assert_eq!(LetterGrade::D.publication_decision(), "Major rework needed");
        assert_eq!(LetterGrade::F.publication_decision(), "Do not publish");
    }

    #[test]
    fn test_letter_grade_is_publishable() {
        assert!(LetterGrade::A.is_publishable());
        assert!(LetterGrade::B.is_publishable());
        assert!(!LetterGrade::C.is_publishable());
        assert!(!LetterGrade::D.is_publishable());
        assert!(!LetterGrade::F.is_publishable());
    }

    #[test]
    fn test_letter_grade_display() {
        assert_eq!(format!("{}", LetterGrade::A), "A");
        assert_eq!(format!("{}", LetterGrade::B), "B");
        assert_eq!(format!("{}", LetterGrade::C), "C");
        assert_eq!(format!("{}", LetterGrade::D), "D");
        assert_eq!(format!("{}", LetterGrade::F), "F");
    }

    #[test]
    fn test_checklist_item_new() {
        let item = ChecklistItem::new(1, "Schema version documented", Severity::Critical, true);
        assert_eq!(item.id, 1);
        assert_eq!(item.description, "Schema version documented");
        assert!(item.passed);
        assert_eq!(item.severity, Severity::Critical);
        assert!(item.suggestion.is_none());
    }

    #[test]
    fn test_checklist_item_with_suggestion() {
        let item = ChecklistItem::new(1, "Schema version documented", Severity::Critical, false)
            .with_suggestion("Add schema_version field to metadata");
        assert!(item.suggestion.is_some());
        assert_eq!(
            item.suggestion.unwrap(),
            "Add schema_version field to metadata"
        );
    }

    #[test]
    fn test_checklist_item_points() {
        let passed_critical = ChecklistItem::new(1, "Test", Severity::Critical, true);
        assert!((passed_critical.points_earned() - 2.0).abs() < 0.01);
        assert!((passed_critical.max_points() - 2.0).abs() < 0.01);

        let failed_critical = ChecklistItem::new(2, "Test", Severity::Critical, false);
        assert!((failed_critical.points_earned() - 0.0).abs() < 0.01);
        assert!((failed_critical.max_points() - 2.0).abs() < 0.01);

        let passed_low = ChecklistItem::new(3, "Test", Severity::Low, true);
        assert!((passed_low.points_earned() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_quality_score_perfect() {
        let checklist = vec![
            ChecklistItem::new(1, "Critical check", Severity::Critical, true),
            ChecklistItem::new(2, "High check", Severity::High, true),
            ChecklistItem::new(3, "Medium check", Severity::Medium, true),
            ChecklistItem::new(4, "Low check", Severity::Low, true),
        ];
        let score = QualityScore::from_checklist(checklist);

        // Total max points: 2.0 + 1.5 + 1.0 + 0.5 = 5.0
        // Total earned: 5.0
        // Score: 100%
        assert!((score.score - 100.0).abs() < 0.01);
        assert_eq!(score.grade, LetterGrade::A);
        assert!(score.grade.is_publishable());
        assert!(!score.has_critical_failures());
    }

    #[test]
    fn test_quality_score_with_critical_failure() {
        let checklist = vec![
            ChecklistItem::new(1, "Critical check", Severity::Critical, false),
            ChecklistItem::new(2, "High check", Severity::High, true),
            ChecklistItem::new(3, "Medium check", Severity::Medium, true),
            ChecklistItem::new(4, "Low check", Severity::Low, true),
        ];
        let score = QualityScore::from_checklist(checklist);

        // Total max: 5.0, Earned: 3.0, Score: 60%
        assert!((score.score - 60.0).abs() < 0.01);
        assert_eq!(score.grade, LetterGrade::D);
        assert!(score.has_critical_failures());
        assert!(!score.grade.is_publishable());
    }

    #[test]
    fn test_quality_score_failed_items() {
        let checklist = vec![
            ChecklistItem::new(1, "Critical check", Severity::Critical, false),
            ChecklistItem::new(2, "High check", Severity::High, true),
            ChecklistItem::new(3, "Medium check", Severity::Medium, false),
        ];
        let score = QualityScore::from_checklist(checklist);

        let failed = score.failed_items();
        assert_eq!(failed.len(), 2);
        assert_eq!(failed[0].id, 1);
        assert_eq!(failed[1].id, 3);

        let critical = score.critical_failures();
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].id, 1);
    }

    #[test]
    fn test_quality_score_severity_breakdown() {
        let checklist = vec![
            ChecklistItem::new(1, "C1", Severity::Critical, true),
            ChecklistItem::new(2, "C2", Severity::Critical, false),
            ChecklistItem::new(3, "H1", Severity::High, true),
        ];
        let score = QualityScore::from_checklist(checklist);

        let critical_stats = score.severity_breakdown.get(&Severity::Critical).unwrap();
        assert_eq!(critical_stats.total, 2);
        assert_eq!(critical_stats.passed, 1);
        assert_eq!(critical_stats.failed, 1);

        let high_stats = score.severity_breakdown.get(&Severity::High).unwrap();
        assert_eq!(high_stats.total, 1);
        assert_eq!(high_stats.passed, 1);
    }

    #[test]
    fn test_quality_score_badge_url() {
        let checklist = vec![ChecklistItem::new(1, "Test", Severity::Critical, true)];
        let score = QualityScore::from_checklist(checklist);

        let badge = score.badge_url();
        assert!(badge.contains("shields.io"));
        assert!(badge.contains("data_quality"));
        assert!(badge.contains("brightgreen")); // Grade A
    }

    #[test]
    fn test_quality_score_badge_colors() {
        // Test each grade gets correct color
        let grades_colors = vec![
            (100.0, "brightgreen"), // A
            (90.0, "green"),        // B
            (75.0, "yellow"),       // C
            (55.0, "orange"),       // D
            (30.0, "red"),          // F
        ];

        for (target_score, expected_color) in grades_colors {
            // Create checklist that produces approximately the target score
            let target: f64 = target_score;
            #[allow(clippy::cast_sign_loss)] // target is always positive (30.0-100.0)
            let passed = (target / 100.0 * 10.0).round() as usize;
            let failed = 10 - passed;
            let mut checklist: Vec<ChecklistItem> = (0..passed)
                .map(|i| ChecklistItem::new(i as u8, "Test", Severity::Medium, true))
                .collect();
            checklist.extend((0..failed).map(|i| {
                ChecklistItem::new((passed + i) as u8, "Test", Severity::Medium, false)
            }));

            let score = QualityScore::from_checklist(checklist);
            let badge = score.badge_url();
            assert!(
                badge.contains(expected_color),
                "Score {:.0} should have color {} but badge was {}",
                score.score,
                expected_color,
                badge
            );
        }
    }

    #[test]
    fn test_quality_score_json_output() {
        let checklist = vec![
            ChecklistItem::new(1, "Schema check", Severity::Critical, true),
            ChecklistItem::new(2, "Column check", Severity::High, false)
                .with_suggestion("Add missing columns"),
        ];
        let score = QualityScore::from_checklist(checklist);

        let json = score.to_json();
        assert!(json.contains("\"score\":"));
        assert!(json.contains("\"grade\":"));
        assert!(json.contains("\"is_publishable\":"));
        assert!(json.contains("\"failed_items\":"));
        assert!(json.contains("\"badge_url\":"));
        assert!(json.contains("Add missing columns"));
    }

    #[test]
    fn test_quality_score_empty_checklist() {
        let checklist: Vec<ChecklistItem> = vec![];
        let score = QualityScore::from_checklist(checklist);

        // Empty checklist = 100% (nothing to fail)
        assert!((score.score - 100.0).abs() < 0.01);
        assert_eq!(score.grade, LetterGrade::A);
    }

    // ========== Quality Profile Tests (GH-10) ==========

    #[test]
    fn test_quality_profile_default() {
        let profile = QualityProfile::default();
        assert_eq!(profile.name, "default");
        assert!(profile.expected_constant_columns.is_empty());
        assert!(profile.nullable_columns.is_empty());
        assert!((profile.max_null_ratio - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_quality_profile_doctest_corpus() {
        let profile = QualityProfile::doctest_corpus();
        assert_eq!(profile.name, "doctest-corpus");
        assert!(profile.is_expected_constant("source"));
        assert!(profile.is_expected_constant("version"));
        assert!(!profile.is_expected_constant("function"));
        assert!(profile.is_nullable("signature"));
        assert!(!profile.is_nullable("input"));
    }

    #[test]
    fn test_quality_profile_ml_training() {
        let profile = QualityProfile::ml_training();
        assert_eq!(profile.name, "ml-training");
        assert!((profile.max_null_ratio - 0.0).abs() < 0.001);
        assert!((profile.max_duplicate_ratio - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_quality_profile_time_series() {
        let profile = QualityProfile::time_series();
        assert_eq!(profile.name, "time-series");
        assert!((profile.max_duplicate_row_ratio - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_profile_by_name() {
        assert!(QualityProfile::by_name("default").is_some());
        assert!(QualityProfile::by_name("doctest-corpus").is_some());
        assert!(QualityProfile::by_name("doctest").is_some());
        assert!(QualityProfile::by_name("ml-training").is_some());
        assert!(QualityProfile::by_name("ml").is_some());
        assert!(QualityProfile::by_name("time-series").is_some());
        assert!(QualityProfile::by_name("timeseries").is_some());
        assert!(QualityProfile::by_name("nonexistent").is_none());
    }

    #[test]
    fn test_quality_profile_available_profiles() {
        let profiles = QualityProfile::available_profiles();
        assert!(profiles.contains(&"default"));
        assert!(profiles.contains(&"doctest-corpus"));
        assert!(profiles.contains(&"ml-training"));
        assert!(profiles.contains(&"time-series"));
    }

    #[test]
    fn test_quality_profile_builders() {
        let profile = QualityProfile::new("custom")
            .with_description("Custom profile")
            .with_expected_constant("id")
            .with_nullable("optional_field")
            .with_max_null_ratio(0.2)
            .with_max_duplicate_ratio(0.6);

        assert_eq!(profile.name, "custom");
        assert_eq!(profile.description, "Custom profile");
        assert!(profile.is_expected_constant("id"));
        assert!(profile.is_nullable("optional_field"));
        assert!((profile.max_null_ratio - 0.2).abs() < 0.001);
        assert!((profile.max_duplicate_ratio - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_quality_profile_null_threshold_for() {
        let profile = QualityProfile::doctest_corpus();

        // Nullable columns get 100% threshold
        assert!((profile.null_threshold_for("signature") - 1.0).abs() < 0.001);

        // Non-nullable columns get profile threshold
        assert!((profile.null_threshold_for("input") - profile.max_null_ratio).abs() < 0.001);
    }

    #[test]
    fn test_quality_profile_clone() {
        let profile = QualityProfile::doctest_corpus();
        let cloned = profile.clone();
        assert_eq!(profile.name, cloned.name);
        assert_eq!(
            profile.expected_constant_columns,
            cloned.expected_constant_columns
        );
    }

    #[test]
    fn test_quality_profile_debug() {
        let profile = QualityProfile::default();
        let debug = format!("{:?}", profile);
        assert!(debug.contains("QualityProfile"));
        assert!(debug.contains("default"));
    }
}
