//! Imbalanced dataset detection for ML pipelines
//!
//! Detects class imbalance in classification datasets and provides
//! recommendations for handling strategies.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::imbalance::ImbalanceDetector;
//!
//! let detector = ImbalanceDetector::new("label");
//! let report = detector.analyze(&dataset)?;
//!
//! if report.is_imbalanced() {
//!     println!("Imbalance ratio: {:.2}", report.metrics.imbalance_ratio);
//!     for rec in &report.recommendations {
//!         println!("Recommendation: {}", rec);
//!     }
//! }
//! ```

// Statistical computation requires usize->f64 casts
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::{Error, Result},
};

/// Severity of class imbalance
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImbalanceSeverity {
    /// Balanced dataset (ratio < 1.5)
    None,
    /// Slight imbalance (1.5 <= ratio < 3)
    Low,
    /// Moderate imbalance (3 <= ratio < 10)
    Moderate,
    /// Severe imbalance (10 <= ratio < 100)
    Severe,
    /// Extreme imbalance (ratio >= 100)
    Extreme,
}

impl ImbalanceSeverity {
    /// Create severity from imbalance ratio (majority/minority)
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio < 1.5 {
            Self::None
        } else if ratio < 3.0 {
            Self::Low
        } else if ratio < 10.0 {
            Self::Moderate
        } else if ratio < 100.0 {
            Self::Severe
        } else {
            Self::Extreme
        }
    }

    /// Check if this represents actual imbalance
    pub fn is_imbalanced(&self) -> bool {
        *self != Self::None
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "Balanced",
            Self::Low => "Slightly imbalanced",
            Self::Moderate => "Moderately imbalanced",
            Self::Severe => "Severely imbalanced",
            Self::Extreme => "Extremely imbalanced",
        }
    }
}

/// Metrics for measuring class imbalance
#[derive(Debug, Clone)]
pub struct ImbalanceMetrics {
    /// Ratio of majority to minority class (>= 1.0)
    pub imbalance_ratio: f64,
    /// Shannon entropy of class distribution (0 = single class, log(n) =
    /// uniform)
    pub entropy: f64,
    /// Normalized entropy (0-1, 1 = perfectly balanced)
    pub normalized_entropy: f64,
    /// Gini impurity (0 = single class, 1-1/n = uniform)
    pub gini: f64,
    /// Severity classification
    pub severity: ImbalanceSeverity,
}

impl ImbalanceMetrics {
    /// Create metrics from class counts
    pub fn from_counts(counts: &HashMap<String, usize>) -> Self {
        if counts.is_empty() {
            return Self {
                imbalance_ratio: 1.0,
                entropy: 0.0,
                normalized_entropy: 1.0,
                gini: 0.0,
                severity: ImbalanceSeverity::None,
            };
        }

        let total: usize = counts.values().sum();
        if total == 0 {
            return Self {
                imbalance_ratio: 1.0,
                entropy: 0.0,
                normalized_entropy: 1.0,
                gini: 0.0,
                severity: ImbalanceSeverity::None,
            };
        }

        let total_f = total as f64;
        let n_classes = counts.len();

        // Imbalance ratio
        let max_count = counts.values().copied().max().unwrap_or(0);
        let min_count = counts.values().copied().min().unwrap_or(0);
        let imbalance_ratio = if min_count > 0 {
            max_count as f64 / min_count as f64
        } else {
            f64::INFINITY
        };

        // Shannon entropy: -sum(p * log(p))
        let entropy: f64 = counts
            .values()
            .map(|&c| {
                if c > 0 {
                    let p = c as f64 / total_f;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // Normalized entropy (relative to maximum possible)
        let max_entropy = (n_classes as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            1.0
        };

        // Gini impurity: 1 - sum(p^2)
        let gini: f64 = 1.0
            - counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total_f;
                    p * p
                })
                .sum::<f64>();

        let severity = ImbalanceSeverity::from_ratio(imbalance_ratio);

        Self {
            imbalance_ratio,
            entropy,
            normalized_entropy,
            gini,
            severity,
        }
    }

    /// Check if the dataset is imbalanced
    pub fn is_imbalanced(&self) -> bool {
        self.severity.is_imbalanced()
    }
}

/// Distribution of classes in a dataset
#[derive(Debug, Clone)]
pub struct ClassDistribution {
    /// Count per class
    pub counts: HashMap<String, usize>,
    /// Proportion per class (0-1)
    pub proportions: HashMap<String, f64>,
    /// Total number of samples
    pub total: usize,
    /// Number of unique classes
    pub num_classes: usize,
    /// Majority class name
    pub majority_class: Option<String>,
    /// Minority class name
    pub minority_class: Option<String>,
}

impl ClassDistribution {
    /// Create distribution from class counts
    pub fn from_counts(counts: HashMap<String, usize>) -> Self {
        let total: usize = counts.values().sum();
        let num_classes = counts.len();

        let proportions: HashMap<String, f64> = counts
            .iter()
            .map(|(k, &v)| {
                let prop = if total > 0 {
                    v as f64 / total as f64
                } else {
                    0.0
                };
                (k.clone(), prop)
            })
            .collect();

        let majority_class = counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.clone());

        let minority_class = counts
            .iter()
            .filter(|(_, &v)| v > 0)
            .min_by_key(|(_, &v)| v)
            .map(|(k, _)| k.clone());

        Self {
            counts,
            proportions,
            total,
            num_classes,
            majority_class,
            minority_class,
        }
    }

    /// Get count for a specific class
    pub fn get_count(&self, class: &str) -> usize {
        self.counts.get(class).copied().unwrap_or(0)
    }

    /// Get proportion for a specific class
    pub fn get_proportion(&self, class: &str) -> f64 {
        self.proportions.get(class).copied().unwrap_or(0.0)
    }
}

/// Recommendation for handling imbalanced data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImbalanceRecommendation {
    /// No action needed
    NoAction,
    /// Use stratified sampling for train/test splits
    UseStratifiedSplit,
    /// Consider class weights in model training
    UseClassWeights,
    /// Consider oversampling minority class
    ConsiderOversampling,
    /// Consider undersampling majority class
    ConsiderUndersampling,
    /// Consider SMOTE or similar synthetic generation
    ConsiderSMOTE,
    /// Collect more data for minority classes
    CollectMoreData,
    /// Use appropriate metrics (F1, AUC-ROC, not accuracy)
    UseAppropriateMetrics,
    /// Consider anomaly detection approach
    ConsiderAnomalyDetection,
}

impl ImbalanceRecommendation {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::NoAction => "No action needed - dataset is balanced",
            Self::UseStratifiedSplit => "Use stratified sampling for train/test splits",
            Self::UseClassWeights => "Apply class weights during model training",
            Self::ConsiderOversampling => "Consider oversampling the minority class",
            Self::ConsiderUndersampling => "Consider undersampling the majority class",
            Self::ConsiderSMOTE => "Consider SMOTE or synthetic minority oversampling",
            Self::CollectMoreData => "Collect more samples for minority classes",
            Self::UseAppropriateMetrics => {
                "Use F1-score, AUC-ROC, or precision-recall instead of accuracy"
            }
            Self::ConsiderAnomalyDetection => "Consider framing as anomaly detection problem",
        }
    }
}

impl std::fmt::Display for ImbalanceRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Report from imbalance analysis
#[derive(Debug, Clone)]
pub struct ImbalanceReport {
    /// Column analyzed
    pub column: String,
    /// Class distribution
    pub distribution: ClassDistribution,
    /// Imbalance metrics
    pub metrics: ImbalanceMetrics,
    /// Recommendations
    pub recommendations: Vec<ImbalanceRecommendation>,
}

impl ImbalanceReport {
    /// Create report from distribution
    pub fn from_distribution(column: impl Into<String>, distribution: ClassDistribution) -> Self {
        let metrics = ImbalanceMetrics::from_counts(&distribution.counts);
        let recommendations = generate_recommendations(&metrics, &distribution);

        Self {
            column: column.into(),
            distribution,
            metrics,
            recommendations,
        }
    }

    /// Check if the dataset is imbalanced
    pub fn is_imbalanced(&self) -> bool {
        self.metrics.is_imbalanced()
    }

    /// Get severity
    pub fn severity(&self) -> ImbalanceSeverity {
        self.metrics.severity
    }
}

/// Generate recommendations based on metrics
fn generate_recommendations(
    metrics: &ImbalanceMetrics,
    distribution: &ClassDistribution,
) -> Vec<ImbalanceRecommendation> {
    let mut recs = Vec::new();

    match metrics.severity {
        ImbalanceSeverity::None => {
            recs.push(ImbalanceRecommendation::NoAction);
        }
        ImbalanceSeverity::Low => {
            recs.push(ImbalanceRecommendation::UseStratifiedSplit);
            recs.push(ImbalanceRecommendation::UseAppropriateMetrics);
        }
        ImbalanceSeverity::Moderate => {
            recs.push(ImbalanceRecommendation::UseStratifiedSplit);
            recs.push(ImbalanceRecommendation::UseClassWeights);
            recs.push(ImbalanceRecommendation::UseAppropriateMetrics);
            if distribution.total < 10000 {
                recs.push(ImbalanceRecommendation::ConsiderOversampling);
            } else {
                recs.push(ImbalanceRecommendation::ConsiderUndersampling);
            }
        }
        ImbalanceSeverity::Severe => {
            recs.push(ImbalanceRecommendation::UseStratifiedSplit);
            recs.push(ImbalanceRecommendation::UseClassWeights);
            recs.push(ImbalanceRecommendation::ConsiderSMOTE);
            recs.push(ImbalanceRecommendation::UseAppropriateMetrics);
            recs.push(ImbalanceRecommendation::CollectMoreData);
        }
        ImbalanceSeverity::Extreme => {
            recs.push(ImbalanceRecommendation::ConsiderAnomalyDetection);
            recs.push(ImbalanceRecommendation::UseStratifiedSplit);
            recs.push(ImbalanceRecommendation::ConsiderSMOTE);
            recs.push(ImbalanceRecommendation::CollectMoreData);
            recs.push(ImbalanceRecommendation::UseAppropriateMetrics);
        }
    }

    recs
}

/// Detector for class imbalance in datasets
pub struct ImbalanceDetector {
    /// Label column name
    label_column: String,
}

impl ImbalanceDetector {
    /// Create a new imbalance detector
    pub fn new(label_column: impl Into<String>) -> Self {
        Self {
            label_column: label_column.into(),
        }
    }

    /// Get the label column name
    pub fn label_column(&self) -> &str {
        &self.label_column
    }

    /// Analyze a dataset for class imbalance
    pub fn analyze(&self, dataset: &ArrowDataset) -> Result<ImbalanceReport> {
        let counts = self.count_classes(dataset)?;
        let distribution = ClassDistribution::from_counts(counts);
        Ok(ImbalanceReport::from_distribution(
            &self.label_column,
            distribution,
        ))
    }

    /// Count class occurrences
    fn count_classes(&self, dataset: &ArrowDataset) -> Result<HashMap<String, usize>> {
        use arrow::array::{Array, Int32Array, Int64Array, StringArray};

        let schema = dataset.schema();
        let col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == &self.label_column)
            .ok_or_else(|| {
                Error::invalid_config(format!(
                    "Column '{}' not found in schema",
                    self.label_column
                ))
            })?;

        let mut counts: HashMap<String, usize> = HashMap::new();

        for batch in dataset.iter() {
            let array = batch.column(col_idx);

            // Handle different array types
            if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let key = arr.value(i).to_string();
                        *counts.entry(key).or_insert(0) += 1;
                    }
                }
            } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let key = arr.value(i).to_string();
                        *counts.entry(key).or_insert(0) += 1;
                    }
                }
            } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let key = arr.value(i).to_string();
                        *counts.entry(key).or_insert(0) += 1;
                    }
                }
            } else {
                return Err(Error::invalid_config(format!(
                    "Unsupported column type for '{}'. Expected string or integer.",
                    self.label_column
                )));
            }
        }

        if counts.is_empty() {
            return Err(Error::invalid_config(format!(
                "No valid values found in column '{}'",
                self.label_column
            )));
        }

        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    // ========== ImbalanceSeverity tests ==========

    #[test]
    fn test_severity_from_ratio() {
        assert_eq!(ImbalanceSeverity::from_ratio(1.0), ImbalanceSeverity::None);
        assert_eq!(ImbalanceSeverity::from_ratio(1.4), ImbalanceSeverity::None);
        assert_eq!(ImbalanceSeverity::from_ratio(1.5), ImbalanceSeverity::Low);
        assert_eq!(ImbalanceSeverity::from_ratio(2.9), ImbalanceSeverity::Low);
        assert_eq!(
            ImbalanceSeverity::from_ratio(3.0),
            ImbalanceSeverity::Moderate
        );
        assert_eq!(
            ImbalanceSeverity::from_ratio(9.9),
            ImbalanceSeverity::Moderate
        );
        assert_eq!(
            ImbalanceSeverity::from_ratio(10.0),
            ImbalanceSeverity::Severe
        );
        assert_eq!(
            ImbalanceSeverity::from_ratio(99.0),
            ImbalanceSeverity::Severe
        );
        assert_eq!(
            ImbalanceSeverity::from_ratio(100.0),
            ImbalanceSeverity::Extreme
        );
        assert_eq!(
            ImbalanceSeverity::from_ratio(1000.0),
            ImbalanceSeverity::Extreme
        );
    }

    #[test]
    fn test_severity_is_imbalanced() {
        assert!(!ImbalanceSeverity::None.is_imbalanced());
        assert!(ImbalanceSeverity::Low.is_imbalanced());
        assert!(ImbalanceSeverity::Moderate.is_imbalanced());
        assert!(ImbalanceSeverity::Severe.is_imbalanced());
        assert!(ImbalanceSeverity::Extreme.is_imbalanced());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ImbalanceSeverity::None < ImbalanceSeverity::Low);
        assert!(ImbalanceSeverity::Low < ImbalanceSeverity::Moderate);
        assert!(ImbalanceSeverity::Moderate < ImbalanceSeverity::Severe);
        assert!(ImbalanceSeverity::Severe < ImbalanceSeverity::Extreme);
    }

    #[test]
    fn test_severity_description() {
        assert_eq!(ImbalanceSeverity::None.description(), "Balanced");
        assert_eq!(
            ImbalanceSeverity::Extreme.description(),
            "Extremely imbalanced"
        );
    }

    // ========== ImbalanceMetrics tests ==========

    #[test]
    fn test_metrics_balanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 100), ("B".to_string(), 100)]
            .into_iter()
            .collect();

        let metrics = ImbalanceMetrics::from_counts(&counts);

        assert!((metrics.imbalance_ratio - 1.0).abs() < 0.01);
        assert!((metrics.normalized_entropy - 1.0).abs() < 0.01);
        assert!((metrics.gini - 0.5).abs() < 0.01);
        assert_eq!(metrics.severity, ImbalanceSeverity::None);
        assert!(!metrics.is_imbalanced());
    }

    #[test]
    fn test_metrics_imbalanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 900), ("B".to_string(), 100)]
            .into_iter()
            .collect();

        let metrics = ImbalanceMetrics::from_counts(&counts);

        assert!((metrics.imbalance_ratio - 9.0).abs() < 0.01);
        assert!(metrics.normalized_entropy < 0.8);
        assert_eq!(metrics.severity, ImbalanceSeverity::Moderate);
        assert!(metrics.is_imbalanced());
    }

    #[test]
    fn test_metrics_severely_imbalanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 990), ("B".to_string(), 10)]
            .into_iter()
            .collect();

        let metrics = ImbalanceMetrics::from_counts(&counts);

        assert!((metrics.imbalance_ratio - 99.0).abs() < 0.01);
        assert_eq!(metrics.severity, ImbalanceSeverity::Severe);
    }

    #[test]
    fn test_metrics_empty() {
        let counts: HashMap<String, usize> = HashMap::new();
        let metrics = ImbalanceMetrics::from_counts(&counts);

        assert!((metrics.imbalance_ratio - 1.0).abs() < 0.01);
        assert_eq!(metrics.severity, ImbalanceSeverity::None);
    }

    #[test]
    fn test_metrics_single_class() {
        let counts: HashMap<String, usize> = [("A".to_string(), 100)].into_iter().collect();

        let metrics = ImbalanceMetrics::from_counts(&counts);

        // Single class has infinite imbalance ratio (no minority)
        assert!(metrics.imbalance_ratio.is_infinite() || metrics.imbalance_ratio >= 1.0);
        assert!((metrics.entropy - 0.0).abs() < 0.01);
        assert!((metrics.gini - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_multiclass_balanced() {
        let counts: HashMap<String, usize> = [
            ("A".to_string(), 100),
            ("B".to_string(), 100),
            ("C".to_string(), 100),
        ]
        .into_iter()
        .collect();

        let metrics = ImbalanceMetrics::from_counts(&counts);

        assert!((metrics.imbalance_ratio - 1.0).abs() < 0.01);
        assert!((metrics.normalized_entropy - 1.0).abs() < 0.01);
    }

    // ========== ClassDistribution tests ==========

    #[test]
    fn test_distribution_from_counts() {
        let counts: HashMap<String, usize> = [("A".to_string(), 75), ("B".to_string(), 25)]
            .into_iter()
            .collect();

        let dist = ClassDistribution::from_counts(counts);

        assert_eq!(dist.total, 100);
        assert_eq!(dist.num_classes, 2);
        assert_eq!(dist.get_count("A"), 75);
        assert_eq!(dist.get_count("B"), 25);
        assert!((dist.get_proportion("A") - 0.75).abs() < 0.01);
        assert!((dist.get_proportion("B") - 0.25).abs() < 0.01);
        assert_eq!(dist.majority_class, Some("A".to_string()));
        assert_eq!(dist.minority_class, Some("B".to_string()));
    }

    #[test]
    fn test_distribution_missing_class() {
        let counts: HashMap<String, usize> = [("A".to_string(), 100)].into_iter().collect();
        let dist = ClassDistribution::from_counts(counts);

        assert_eq!(dist.get_count("B"), 0);
        assert!((dist.get_proportion("B") - 0.0).abs() < 0.01);
    }

    // ========== ImbalanceRecommendation tests ==========

    #[test]
    fn test_recommendation_display() {
        let rec = ImbalanceRecommendation::UseStratifiedSplit;
        assert!(rec.to_string().contains("stratified"));
    }

    // ========== ImbalanceReport tests ==========

    #[test]
    fn test_report_balanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 100), ("B".to_string(), 100)]
            .into_iter()
            .collect();
        let dist = ClassDistribution::from_counts(counts);
        let report = ImbalanceReport::from_distribution("label", dist);

        assert!(!report.is_imbalanced());
        assert_eq!(report.severity(), ImbalanceSeverity::None);
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::NoAction));
    }

    #[test]
    fn test_report_imbalanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 900), ("B".to_string(), 100)]
            .into_iter()
            .collect();
        let dist = ClassDistribution::from_counts(counts);
        let report = ImbalanceReport::from_distribution("label", dist);

        assert!(report.is_imbalanced());
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::UseStratifiedSplit));
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::UseAppropriateMetrics));
    }

    #[test]
    fn test_report_severely_imbalanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 9900), ("B".to_string(), 100)]
            .into_iter()
            .collect();
        let dist = ClassDistribution::from_counts(counts);
        let report = ImbalanceReport::from_distribution("label", dist);

        assert_eq!(report.severity(), ImbalanceSeverity::Severe);
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::ConsiderSMOTE));
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::CollectMoreData));
    }

    #[test]
    fn test_report_extremely_imbalanced() {
        let counts: HashMap<String, usize> = [("A".to_string(), 10000), ("B".to_string(), 10)]
            .into_iter()
            .collect();
        let dist = ClassDistribution::from_counts(counts);
        let report = ImbalanceReport::from_distribution("label", dist);

        assert_eq!(report.severity(), ImbalanceSeverity::Extreme);
        assert!(report
            .recommendations
            .contains(&ImbalanceRecommendation::ConsiderAnomalyDetection));
    }

    // ========== ImbalanceDetector tests ==========

    fn make_string_dataset(labels: Vec<&str>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "label",
            DataType::Utf8,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(labels))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    fn make_int_dataset(labels: Vec<i32>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "label",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(labels))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    #[test]
    fn test_detector_new() {
        let detector = ImbalanceDetector::new("label");
        assert_eq!(detector.label_column(), "label");
    }

    #[test]
    fn test_detector_analyze_balanced() {
        let labels: Vec<&str> = (0..100).map(|i| if i < 50 { "A" } else { "B" }).collect();
        let dataset = make_string_dataset(labels);

        let detector = ImbalanceDetector::new("label");
        let report = detector.analyze(&dataset).expect("analyze");

        assert!(!report.is_imbalanced());
        assert_eq!(report.distribution.total, 100);
        assert_eq!(report.distribution.num_classes, 2);
    }

    #[test]
    fn test_detector_analyze_imbalanced() {
        let mut labels: Vec<&str> = vec!["A"; 90];
        labels.extend(vec!["B"; 10]);
        let dataset = make_string_dataset(labels);

        let detector = ImbalanceDetector::new("label");
        let report = detector.analyze(&dataset).expect("analyze");

        assert!(report.is_imbalanced());
        assert_eq!(report.distribution.majority_class, Some("A".to_string()));
        assert_eq!(report.distribution.minority_class, Some("B".to_string()));
    }

    #[test]
    fn test_detector_analyze_int_labels() {
        let labels: Vec<i32> = (0..100).map(|i| if i < 80 { 0 } else { 1 }).collect();
        let dataset = make_int_dataset(labels);

        let detector = ImbalanceDetector::new("label");
        let report = detector.analyze(&dataset).expect("analyze");

        assert!(report.is_imbalanced());
        assert_eq!(report.distribution.get_count("0"), 80);
        assert_eq!(report.distribution.get_count("1"), 20);
    }

    #[test]
    fn test_detector_missing_column() {
        let dataset = make_string_dataset(vec!["A", "B", "A"]);

        let detector = ImbalanceDetector::new("nonexistent");
        let result = detector.analyze(&dataset);

        assert!(result.is_err());
    }

    #[test]
    fn test_detector_multiclass() {
        let mut labels = vec!["A"; 50];
        labels.extend(vec!["B"; 30]);
        labels.extend(vec!["C"; 20]);
        let dataset = make_string_dataset(labels);

        let detector = ImbalanceDetector::new("label");
        let report = detector.analyze(&dataset).expect("analyze");

        assert_eq!(report.distribution.num_classes, 3);
        assert_eq!(report.distribution.majority_class, Some("A".to_string()));
        assert_eq!(report.distribution.minority_class, Some("C".to_string()));
    }
}
