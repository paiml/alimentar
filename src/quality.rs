//! Data quality assessment for ML pipelines
//!
//! Detects data quality issues including missing values, outliers,
//! duplicates, and schema problems.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::quality::QualityChecker;
//!
//! let checker = QualityChecker::new()
//!     .max_null_ratio(0.1)
//!     .max_duplicate_ratio(0.05);
//!
//! let report = checker.check(&dataset)?;
//! for issue in &report.issues {
//!     println!("Issue: {:?}", issue);
//! }
//! ```

// Statistical computation and internal methods
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::unused_self)]
#![allow(clippy::if_not_else)]

use std::collections::{HashMap, HashSet};

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::Result,
};

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
}
