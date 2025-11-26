//! Data transforms for alimentar.
//!
//! Transforms apply operations to RecordBatches, enabling data preprocessing
//! pipelines. All transforms are composable and can be chained together.

use std::sync::Arc;

use arrow::{
    array::{Array, BooleanArray, RecordBatch},
    compute::filter_record_batch,
    datatypes::{Field, Schema},
};
#[cfg(feature = "shuffle")]
use rand::{seq::SliceRandom, SeedableRng};

use crate::error::{Error, Result};

/// A transform that can be applied to RecordBatches.
///
/// Transforms are the building blocks for data preprocessing pipelines.
/// They take a RecordBatch and produce a new RecordBatch with the
/// transformation applied.
///
/// # Thread Safety
///
/// All transforms must be thread-safe (Send + Sync) to support parallel
/// data loading in future versions.
pub trait Transform: Send + Sync {
    /// Applies the transform to a RecordBatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform cannot be applied to the batch.
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

/// A transform that applies a function to each RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Map;
///
/// let transform = Map::new(|batch| {
///     // Process batch
///     Ok(batch)
/// });
/// ```
pub struct Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    func: F,
}

impl<F> Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    /// Creates a new Map transform with the given function.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Transform for Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (self.func)(batch)
    }
}

/// A transform that filters rows based on a predicate.
///
/// The predicate function receives a RecordBatch and must return a BooleanArray
/// with the same number of rows, where `true` indicates the row should be kept.
///
/// # Example
///
/// ```ignore
/// use alimentar::Filter;
/// use arrow::array::{Int32Array, BooleanArray};
///
/// let filter = Filter::new(|batch| {
///     let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
///     let mask: Vec<bool> = (0..col.len()).map(|i| col.value(i) > 5).collect();
///     Ok(BooleanArray::from(mask))
/// });
/// ```
pub struct Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    predicate: F,
}

impl<F> Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    /// Creates a new Filter transform with the given predicate.
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> Transform for Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mask = (self.predicate)(&batch)?;
        filter_record_batch(&batch, &mask).map_err(Error::Arrow)
    }
}

/// A transform that selects specific columns from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Select;
///
/// let select = Select::new(vec!["id", "name"]);
/// ```
#[derive(Debug, Clone)]
pub struct Select {
    columns: Vec<String>,
}

impl Select {
    /// Creates a new Select transform for the given column names.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the columns to be selected.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}

impl Transform for Select {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(self.columns.len());
        let mut arrays = Vec::with_capacity(self.columns.len());

        for col_name in &self.columns {
            let (idx, field) = schema
                .column_with_name(col_name)
                .ok_or_else(|| Error::column_not_found(col_name))?;

            fields.push(field.clone());
            arrays.push(Arc::clone(batch.column(idx)));
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// A transform that shuffles rows in a RecordBatch.
///
/// Requires the `shuffle` feature.
///
/// # Example
///
/// ```ignore
/// use alimentar::Shuffle;
///
/// // Random shuffle
/// let shuffle = Shuffle::new();
///
/// // Deterministic shuffle with seed
/// let shuffle = Shuffle::with_seed(42);
/// ```
#[cfg(feature = "shuffle")]
#[derive(Debug, Clone)]
pub struct Shuffle {
    seed: Option<u64>,
}

#[cfg(feature = "shuffle")]
impl Shuffle {
    /// Creates a new Shuffle transform with random ordering.
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Creates a new Shuffle transform with a fixed seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

#[cfg(feature = "shuffle")]
impl Default for Shuffle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "shuffle")]
impl Transform for Shuffle {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch);
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..num_rows).collect();
        let mut rng = match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);

        // Reorder each column according to shuffled indices
        let schema = batch.schema();
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                let indices_array =
                    arrow::array::UInt64Array::from_iter_values(indices.iter().map(|&i| i as u64));
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// A transform that renames columns in a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Rename;
/// use std::collections::HashMap;
///
/// let mut mapping = HashMap::new();
/// mapping.insert("old_name".to_string(), "new_name".to_string());
/// let rename = Rename::new(mapping);
/// ```
#[derive(Debug, Clone)]
pub struct Rename {
    mapping: std::collections::HashMap<String, String>,
}

impl Rename {
    /// Creates a new Rename transform with the given column mappings.
    pub fn new(mapping: std::collections::HashMap<String, String>) -> Self {
        Self { mapping }
    }

    /// Creates a Rename transform from pairs of (old_name, new_name).
    pub fn from_pairs<S: Into<String>>(pairs: impl IntoIterator<Item = (S, S)>) -> Self {
        let mapping = pairs
            .into_iter()
            .map(|(old, new)| (old.into(), new.into()))
            .collect();
        Self { mapping }
    }
}

impl Transform for Rename {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let new_fields: Vec<Field> = schema
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                match self.mapping.get(name) {
                    Some(new_name) => {
                        Field::new(new_name, field.data_type().clone(), field.is_nullable())
                    }
                    None => field.as_ref().clone(),
                }
            })
            .collect();

        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, batch.columns().to_vec()).map_err(Error::Arrow)
    }
}

/// A chain of transforms applied in sequence.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Chain, Select, Shuffle};
///
/// let chain = Chain::new()
///     .then(Select::new(vec!["id", "value"]))
///     .then(Shuffle::with_seed(42));
/// ```
pub struct Chain {
    transforms: Vec<Box<dyn Transform>>,
}

impl Chain {
    /// Creates a new empty transform chain.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Adds a transform to the chain.
    #[must_use]
    pub fn then<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Returns the number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns true if the chain has no transforms.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Chain {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut result = batch;
        for transform in &self.transforms {
            result = transform.apply(result)?;
        }
        Ok(result)
    }
}

// Implement Transform for boxed transforms
impl Transform for Box<dyn Transform> {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (**self).apply(batch)
    }
}

// Implement Transform for Arc<dyn Transform>
impl Transform for Arc<dyn Transform> {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (**self).apply(batch)
    }
}

/// A transform that drops (removes) specified columns from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Drop;
///
/// let drop = Drop::new(vec!["temp_column", "debug_info"]);
/// ```
#[derive(Debug, Clone)]
pub struct Drop {
    columns: Vec<String>,
}

impl Drop {
    /// Creates a new Drop transform for the given column names.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the columns to be dropped.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}

impl Transform for Drop {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let drop_set: std::collections::HashSet<&str> =
            self.columns.iter().map(String::as_str).collect();

        let mut fields = Vec::new();
        let mut arrays = Vec::new();

        for (idx, field) in schema.fields().iter().enumerate() {
            if !drop_set.contains(field.name().as_str()) {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(batch.column(idx)));
            }
        }

        if fields.is_empty() {
            return Err(Error::transform("Cannot drop all columns from batch"));
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// A transform that casts columns to different data types.
///
/// # Example
///
/// ```ignore
/// use alimentar::Cast;
/// use arrow::datatypes::DataType;
///
/// let cast = Cast::new(vec![
///     ("id", DataType::Int64),
///     ("value", DataType::Float64),
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct Cast {
    mappings: Vec<(String, arrow::datatypes::DataType)>,
}

impl Cast {
    /// Creates a new Cast transform with column-to-type mappings.
    pub fn new<S: Into<String>>(
        mappings: impl IntoIterator<Item = (S, arrow::datatypes::DataType)>,
    ) -> Self {
        Self {
            mappings: mappings
                .into_iter()
                .map(|(name, dtype)| (name.into(), dtype))
                .collect(),
        }
    }

    /// Returns the cast mappings.
    pub fn mappings(&self) -> &[(String, arrow::datatypes::DataType)] {
        &self.mappings
    }
}

impl Transform for Cast {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::compute::cast;

        let schema = batch.schema();
        let cast_map: std::collections::HashMap<&str, &arrow::datatypes::DataType> =
            self.mappings.iter().map(|(n, t)| (n.as_str(), t)).collect();

        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut arrays = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(idx);

            if let Some(&target_type) = cast_map.get(field.name().as_str()) {
                let casted = cast(col.as_ref(), target_type).map_err(|e| {
                    Error::transform(format!(
                        "Failed to cast column '{}' to {:?}: {}",
                        field.name(),
                        target_type,
                        e
                    ))
                })?;
                fields.push(Field::new(
                    field.name(),
                    target_type.clone(),
                    field.is_nullable(),
                ));
                arrays.push(casted);
            } else {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(col));
            }
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// Normalization method for numeric columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMethod {
    /// Min-max normalization: (x - min) / (max - min), scales to [0, 1]
    MinMax,
    /// Z-score normalization: (x - mean) / std, centers around 0
    ZScore,
    /// Scale to unit length (L2 norm)
    L2,
}

/// A transform that normalizes numeric columns.
///
/// Supports min-max scaling, z-score standardization, and L2 normalization.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Normalize, NormMethod};
///
/// // Min-max normalize specific columns
/// let normalize = Normalize::new(vec!["feature1", "feature2"], NormMethod::MinMax);
///
/// // Z-score normalize all numeric columns
/// let normalize = Normalize::all_numeric(NormMethod::ZScore);
/// ```
#[derive(Debug, Clone)]
pub struct Normalize {
    columns: Option<Vec<String>>,
    method: NormMethod,
}

impl Normalize {
    /// Creates a Normalize transform for specific columns.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>, method: NormMethod) -> Self {
        Self {
            columns: Some(columns.into_iter().map(Into::into).collect()),
            method,
        }
    }

    /// Creates a Normalize transform that applies to all numeric columns.
    pub fn all_numeric(method: NormMethod) -> Self {
        Self {
            columns: None,
            method,
        }
    }

    /// Returns the columns to normalize (None means all numeric).
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
    }

    /// Returns the normalization method.
    pub fn method(&self) -> NormMethod {
        self.method
    }

    fn is_numeric_type(dtype: &arrow::datatypes::DataType) -> bool {
        use arrow::datatypes::DataType;
        matches!(
            dtype,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
        )
    }

    fn normalize_array(
        &self,
        array: &dyn Array,
        _dtype: &arrow::datatypes::DataType,
    ) -> Result<Arc<dyn Array>> {
        use arrow::{array::Float64Array, compute::cast, datatypes::DataType};

        // Cast to Float64 for normalization
        let float_array = cast(array, &DataType::Float64)
            .map_err(|e| Error::transform(format!("Failed to cast to Float64: {}", e)))?;

        let float_values = float_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::transform("Expected Float64Array after cast"))?;

        let normalized = match self.method {
            NormMethod::MinMax => Self::min_max_normalize(float_values),
            NormMethod::ZScore => Self::zscore_normalize(float_values),
            NormMethod::L2 => Self::l2_normalize(float_values),
        };

        // Normalized values are always Float64 (they're now in [0,1] or centered)
        Ok(Arc::new(normalized))
    }

    fn min_max_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let v = array.value(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }

        let range = max - min;
        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if range == 0.0 {
                    Some(0.0)
                } else {
                    Some((array.value(i) - min) / range)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }

    #[allow(clippy::cast_precision_loss)]
    fn zscore_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut sum = 0.0;
        let mut count = 0usize;

        for i in 0..array.len() {
            if !array.is_null(i) {
                sum += array.value(i);
                count += 1;
            }
        }

        if count == 0 {
            return array.clone();
        }

        let mean = sum / count as f64;

        let mut variance_sum = 0.0;
        for i in 0..array.len() {
            if !array.is_null(i) {
                let diff = array.value(i) - mean;
                variance_sum += diff * diff;
            }
        }

        let std = (variance_sum / count as f64).sqrt();

        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if std == 0.0 {
                    Some(0.0)
                } else {
                    Some((array.value(i) - mean) / std)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }

    fn l2_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut sum_sq = 0.0;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let v = array.value(i);
                sum_sq += v * v;
            }
        }

        let norm = sum_sq.sqrt();

        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if norm == 0.0 {
                    Some(0.0)
                } else {
                    Some(array.value(i) / norm)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }
}

impl Transform for Normalize {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();

        let columns_to_normalize: std::collections::HashSet<&str> = match &self.columns {
            Some(cols) => cols.iter().map(String::as_str).collect(),
            None => schema
                .fields()
                .iter()
                .filter(|f| Self::is_numeric_type(f.data_type()))
                .map(|f| f.name().as_str())
                .collect(),
        };

        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut arrays = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(idx);

            if columns_to_normalize.contains(field.name().as_str()) {
                if !Self::is_numeric_type(field.data_type()) {
                    return Err(Error::transform(format!(
                        "Column '{}' is not numeric (type: {:?})",
                        field.name(),
                        field.data_type()
                    )));
                }

                let normalized = self.normalize_array(col.as_ref(), field.data_type())?;
                // Normalized columns become Float64
                fields.push(Field::new(
                    field.name(),
                    arrow::datatypes::DataType::Float64,
                    field.is_nullable(),
                ));
                arrays.push(normalized);
            } else {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(col));
            }
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// A transform that randomly samples rows from a RecordBatch.
///
/// Useful for creating train/test splits or reducing dataset size.
/// Requires the `shuffle` feature.
///
/// # Example
///
/// ```ignore
/// use alimentar::Sample;
///
/// // Sample 100 rows with a fixed seed
/// let sample = Sample::new(100).with_seed(42);
///
/// // Sample 10% of rows
/// let sample = Sample::fraction(0.1);
/// ```
#[cfg(feature = "shuffle")]
#[derive(Debug, Clone)]
pub struct Sample {
    count: Option<usize>,
    fraction: Option<f64>,
    seed: Option<u64>,
}

#[cfg(feature = "shuffle")]
impl Sample {
    /// Creates a Sample transform that selects exactly `count` rows.
    ///
    /// If the batch has fewer rows than `count`, all rows are returned.
    pub fn new(count: usize) -> Self {
        Self {
            count: Some(count),
            fraction: None,
            seed: None,
        }
    }

    /// Creates a Sample transform that selects a fraction of rows.
    ///
    /// The fraction should be between 0.0 and 1.0.
    pub fn fraction(frac: f64) -> Self {
        Self {
            count: None,
            fraction: Some(frac.clamp(0.0, 1.0)),
            seed: None,
        }
    }

    /// Sets a seed for reproducible sampling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns the sample count if set.
    pub fn count(&self) -> Option<usize> {
        self.count
    }

    /// Returns the sample fraction if set.
    pub fn sample_fraction(&self) -> Option<f64> {
        self.fraction
    }
}

#[cfg(feature = "shuffle")]
impl Transform for Sample {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(batch);
        }

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let sample_size = match (self.count, self.fraction) {
            (Some(c), _) => c.min(num_rows),
            (None, Some(f)) => ((num_rows as f64) * f).round() as usize,
            (None, None) => return Ok(batch),
        };

        if sample_size >= num_rows {
            return Ok(batch);
        }

        // Create shuffled indices and take first sample_size
        let mut indices: Vec<usize> = (0..num_rows).collect();
        let mut rng = match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);
        indices.sort_unstable(); // Keep original order

        // Reorder each column according to sampled indices
        let schema = batch.schema();
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                let indices_array =
                    arrow::array::UInt64Array::from_iter_values(indices.iter().map(|&i| i as u64));
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// A transform that takes the first N rows from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Take;
///
/// let take = Take::new(100); // Take first 100 rows
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Take {
    count: usize,
}

impl Take {
    /// Creates a Take transform that keeps the first `count` rows.
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// Returns the number of rows to take.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Transform for Take {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if self.count >= num_rows {
            return Ok(batch);
        }

        Ok(batch.slice(0, self.count))
    }
}

/// A transform that skips the first N rows from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Skip;
///
/// let skip = Skip::new(10); // Skip first 10 rows
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Skip {
    count: usize,
}

impl Skip {
    /// Creates a Skip transform that skips the first `count` rows.
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// Returns the number of rows to skip.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Transform for Skip {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if self.count >= num_rows {
            // Skip all rows - return empty batch with same schema
            return Ok(batch.slice(0, 0));
        }

        let remaining = num_rows - self.count;
        Ok(batch.slice(self.count, remaining))
    }
}

/// Sort order for the Sort transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Ascending order (smallest to largest)
    #[default]
    Ascending,
    /// Descending order (largest to smallest)
    Descending,
}

/// A transform that sorts rows by one or more columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Sort, SortOrder};
///
/// // Sort by single column ascending
/// let sort = Sort::by("age");
///
/// // Sort by column descending
/// let sort = Sort::by("score").order(SortOrder::Descending);
///
/// // Sort by multiple columns
/// let sort = Sort::by_columns(vec![("name", SortOrder::Ascending), ("age", SortOrder::Descending)]);
/// ```
#[derive(Debug, Clone)]
pub struct Sort {
    columns: Vec<(String, SortOrder)>,
    nulls_first: bool,
}

impl Sort {
    /// Creates a Sort transform for a single column (ascending by default).
    pub fn by<S: Into<String>>(column: S) -> Self {
        Self {
            columns: vec![(column.into(), SortOrder::Ascending)],
            nulls_first: false,
        }
    }

    /// Creates a Sort transform for multiple columns with specified orders.
    pub fn by_columns<S: Into<String>>(columns: impl IntoIterator<Item = (S, SortOrder)>) -> Self {
        Self {
            columns: columns
                .into_iter()
                .map(|(name, order)| (name.into(), order))
                .collect(),
            nulls_first: false,
        }
    }

    /// Sets the sort order for a single-column sort.
    #[must_use]
    pub fn order(mut self, order: SortOrder) -> Self {
        if let Some((_, o)) = self.columns.first_mut() {
            *o = order;
        }
        self
    }

    /// Sets whether nulls should appear first (default: false, nulls last).
    #[must_use]
    pub fn nulls_first(mut self, nulls_first: bool) -> Self {
        self.nulls_first = nulls_first;
        self
    }

    /// Returns the columns and their sort orders.
    pub fn columns(&self) -> &[(String, SortOrder)] {
        &self.columns
    }
}

impl Transform for Sort {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};

        if batch.num_rows() <= 1 || self.columns.is_empty() {
            return Ok(batch);
        }

        let schema = batch.schema();

        // Build sort columns
        let sort_columns: Vec<SortColumn> = self
            .columns
            .iter()
            .map(|(col_name, order)| {
                let (idx, _) = schema
                    .column_with_name(col_name)
                    .ok_or_else(|| Error::column_not_found(col_name))?;

                Ok(SortColumn {
                    values: Arc::clone(batch.column(idx)),
                    options: Some(SortOptions {
                        descending: *order == SortOrder::Descending,
                        nulls_first: self.nulls_first,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_columns, None).map_err(Error::Arrow)?;

        // Reorder all columns
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                take(col.as_ref(), &indices, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// Strategy for filling null values.
#[derive(Debug, Clone)]
pub enum FillStrategy {
    /// Fill with a constant integer value
    Int(i64),
    /// Fill with a constant float value
    Float(f64),
    /// Fill with a constant string value
    String(String),
    /// Fill with a constant boolean value
    Bool(bool),
    /// Fill with zero (works for numeric types)
    Zero,
    /// Forward fill (use previous non-null value)
    Forward,
    /// Backward fill (use next non-null value)
    Backward,
}

/// A transform that fills null values in specified columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::{FillNull, FillStrategy};
///
/// // Fill nulls in "age" column with 0
/// let fill = FillNull::new("age", FillStrategy::Zero);
///
/// // Fill nulls with a specific value
/// let fill = FillNull::new("score", FillStrategy::Float(0.0));
///
/// // Forward fill (use previous value)
/// let fill = FillNull::new("value", FillStrategy::Forward);
/// ```
#[derive(Debug, Clone)]
pub struct FillNull {
    column: String,
    strategy: FillStrategy,
}

impl FillNull {
    /// Creates a FillNull transform for the specified column.
    pub fn new<S: Into<String>>(column: S, strategy: FillStrategy) -> Self {
        Self {
            column: column.into(),
            strategy,
        }
    }

    /// Creates a FillNull transform that fills with zero.
    pub fn with_zero<S: Into<String>>(column: S) -> Self {
        Self::new(column, FillStrategy::Zero)
    }

    /// Returns the column name.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// Returns the fill strategy.
    pub fn strategy(&self) -> &FillStrategy {
        &self.strategy
    }
}

impl Transform for FillNull {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::datatypes::DataType;

        let schema = batch.schema();
        let (col_idx, field) = schema
            .column_with_name(&self.column)
            .ok_or_else(|| Error::column_not_found(&self.column))?;

        let mut arrays: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        let col = batch.column(col_idx);

        let filled: Arc<dyn Array> = match (field.data_type(), &self.strategy) {
            (DataType::Int32, FillStrategy::Int(v)) => {
                Arc::new(Self::fill_i32_array(col, *v as i32))
            }
            (DataType::Int64, FillStrategy::Int(v)) => Arc::new(Self::fill_i64_array(col, *v)),
            (DataType::Float32, FillStrategy::Float(v)) => {
                Arc::new(Self::fill_f32_array(col, *v as f32))
            }
            (DataType::Float64, FillStrategy::Float(v)) => Arc::new(Self::fill_f64_array(col, *v)),
            (DataType::Int32, FillStrategy::Zero) => Arc::new(Self::fill_i32_array(col, 0)),
            (DataType::Int64, FillStrategy::Zero) => Arc::new(Self::fill_i64_array(col, 0)),
            (DataType::Float32, FillStrategy::Zero) => Arc::new(Self::fill_f32_array(col, 0.0)),
            (DataType::Float64, FillStrategy::Zero) => Arc::new(Self::fill_f64_array(col, 0.0)),
            (DataType::Utf8, FillStrategy::String(s)) => Arc::new(Self::fill_string_array(col, s)),
            (DataType::Boolean, FillStrategy::Bool(b)) => Arc::new(Self::fill_bool_array(col, *b)),
            (DataType::Int32, FillStrategy::Forward) => Arc::new(Self::forward_fill_i32(col)),
            (DataType::Int64, FillStrategy::Forward) => Arc::new(Self::forward_fill_i64(col)),
            (DataType::Float64, FillStrategy::Forward) => Arc::new(Self::forward_fill_f64(col)),
            (DataType::Int32, FillStrategy::Backward) => Arc::new(Self::backward_fill_i32(col)),
            (DataType::Int64, FillStrategy::Backward) => Arc::new(Self::backward_fill_i64(col)),
            (DataType::Float64, FillStrategy::Backward) => Arc::new(Self::backward_fill_f64(col)),
            _ => {
                return Err(Error::transform(format!(
                    "Unsupported type {:?} for fill strategy {:?}",
                    field.data_type(),
                    self.strategy
                )));
            }
        };

        arrays[col_idx] = filled;
        RecordBatch::try_new(schema, arrays).map_err(Error::Arrow)
    }
}

impl FillNull {
    fn fill_i32_array(col: &Arc<dyn Array>, fill_value: i32) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let values: Vec<i32> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn fill_i64_array(col: &Arc<dyn Array>, fill_value: i64) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let values: Vec<i64> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn fill_f32_array(col: &Arc<dyn Array>, fill_value: f32) -> arrow::array::Float32Array {
        use arrow::array::Float32Array;
        let arr = col.as_any().downcast_ref::<Float32Array>();
        if let Some(arr) = arr {
            let values: Vec<f32> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Float32Array::from(values)
        } else {
            Float32Array::from(Vec::<f32>::new())
        }
    }

    fn fill_f64_array(col: &Arc<dyn Array>, fill_value: f64) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let values: Vec<f64> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }

    fn fill_string_array(col: &Arc<dyn Array>, fill_value: &str) -> arrow::array::StringArray {
        use arrow::array::StringArray;
        let arr = col.as_any().downcast_ref::<StringArray>();
        if let Some(arr) = arr {
            let values: Vec<&str> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            StringArray::from(values)
        } else {
            StringArray::from(Vec::<&str>::new())
        }
    }

    fn fill_bool_array(col: &Arc<dyn Array>, fill_value: bool) -> arrow::array::BooleanArray {
        use arrow::array::BooleanArray;
        let arr = col.as_any().downcast_ref::<BooleanArray>();
        if let Some(arr) = arr {
            let values: Vec<bool> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            BooleanArray::from(values)
        } else {
            BooleanArray::from(Vec::<bool>::new())
        }
    }

    fn forward_fill_i32(col: &Arc<dyn Array>) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let mut last: Option<i32> = None;
            let values: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn forward_fill_i64(col: &Arc<dyn Array>) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let mut last: Option<i64> = None;
            let values: Vec<Option<i64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn forward_fill_f64(col: &Arc<dyn Array>) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let mut last: Option<f64> = None;
            let values: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }

    fn backward_fill_i32(col: &Arc<dyn Array>) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let mut next: Option<i32> = None;
            let mut values: Vec<Option<i32>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn backward_fill_i64(col: &Arc<dyn Array>) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let mut next: Option<i64> = None;
            let mut values: Vec<Option<i64>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn backward_fill_f64(col: &Arc<dyn Array>) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let mut next: Option<f64> = None;
            let mut values: Vec<Option<f64>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }
}

/// A transform that removes duplicate rows based on specified columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::Unique;
///
/// // Keep only unique rows based on all columns
/// let unique = Unique::all();
///
/// // Keep only unique rows based on specific columns
/// let unique = Unique::by(vec!["user_id", "date"]);
///
/// // Keep first occurrence (default) or last
/// let unique = Unique::by(vec!["id"]).keep_first();
/// let unique = Unique::by(vec!["id"]).keep_last();
/// ```
#[derive(Debug, Clone)]
pub struct Unique {
    columns: Option<Vec<String>>,
    keep_last: bool,
}

impl Unique {
    /// Creates a Unique transform that considers all columns.
    pub fn all() -> Self {
        Self {
            columns: None,
            keep_last: false,
        }
    }

    /// Creates a Unique transform that considers specific columns.
    pub fn by<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: Some(columns.into_iter().map(Into::into).collect()),
            keep_last: false,
        }
    }

    /// Keep the first occurrence of duplicates (default).
    #[must_use]
    pub fn keep_first(mut self) -> Self {
        self.keep_last = false;
        self
    }

    /// Keep the last occurrence of duplicates.
    #[must_use]
    pub fn keep_last(mut self) -> Self {
        self.keep_last = true;
        self
    }

    /// Returns the columns used for uniqueness check.
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
    }
}

impl Transform for Unique {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use std::collections::HashMap;

        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch);
        }

        let schema = batch.schema();

        // Determine which columns to use for uniqueness
        let key_indices: Vec<usize> = match &self.columns {
            Some(cols) => cols
                .iter()
                .map(|name| {
                    schema
                        .column_with_name(name)
                        .map(|(idx, _)| idx)
                        .ok_or_else(|| Error::column_not_found(name))
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..schema.fields().len()).collect(),
        };

        // Build a hash of each row's key columns
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut keep_indices: Vec<usize> = Vec::new();

        let row_iter: Box<dyn Iterator<Item = usize>> = if self.keep_last {
            Box::new((0..num_rows).rev())
        } else {
            Box::new(0..num_rows)
        };

        for row_idx in row_iter {
            // Create a string key from the key columns
            let key = Self::row_key(&batch, row_idx, &key_indices);

            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
                e.insert(row_idx);
                keep_indices.push(row_idx);
            }
        }

        if self.keep_last {
            keep_indices.reverse();
        }

        if keep_indices.len() == num_rows {
            return Ok(batch);
        }

        // Build new batch with only unique rows
        let indices_array =
            arrow::array::UInt64Array::from_iter_values(keep_indices.iter().map(|&i| i as u64));

        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

impl Unique {
    fn row_key(batch: &RecordBatch, row_idx: usize, key_indices: &[usize]) -> String {
        use arrow::array::{
            BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
        };

        let mut parts: Vec<String> = Vec::with_capacity(key_indices.len());

        for &col_idx in key_indices {
            let col = batch.column(col_idx);
            let val = if col.is_null(row_idx) {
                "NULL".to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                arr.value(row_idx).to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                arr.value(row_idx).to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                // Use bits for exact comparison
                arr.value(row_idx).to_bits().to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                arr.value(row_idx).to_bits().to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                arr.value(row_idx).to_string()
            } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                arr.value(row_idx).to_string()
            } else {
                format!("{:?}", col.data_type())
            };
            parts.push(val);
        }

        parts.join("\x00")
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::redundant_closure
)]
mod tests {
    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::DataType,
    };

    use super::*;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let name_array = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let value_array = Int32Array::from(vec![10, 20, 30, 40, 50]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(value_array),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"))
    }

    #[test]
    fn test_map_transform() {
        let batch = create_test_batch();
        let transform = Map::new(|b| Ok(b)); // Identity transform

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_filter_transform() {
        let batch = create_test_batch();
        let transform = Filter::new(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::transform("Expected Int32Array"))?;
            let mask: Vec<bool> = (0..col.len()).map(|i| col.value(i) > 2).collect();
            Ok(BooleanArray::from(mask))
        });

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // Only id > 2: 3, 4, 5
    }

    #[test]
    fn test_select_transform() {
        let batch = create_test_batch();
        let transform = Select::new(vec!["id", "value"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "value");
    }

    #[test]
    fn test_select_column_not_found() {
        let batch = create_test_batch();
        let transform = Select::new(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_shuffle_transform_deterministic() {
        let batch = create_test_batch();
        let transform = Shuffle::with_seed(42);

        let result1 = transform.apply(batch.clone());
        let result2 = transform.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        // Same seed should produce same shuffle
        let col1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let col2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..col1.len() {
            assert_eq!(col1.value(i), col2.value(i));
        }
    }

    #[test]
    fn test_shuffle_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Shuffle::with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        // Check that id-name-value relationships are preserved
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Original: id=1 -> value=10, id=2 -> value=20, etc.
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }
    }

    #[test]
    fn test_rename_transform() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("id", "identifier"), ("name", "label")]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.schema().field(0).name(), "identifier");
        assert_eq!(result.schema().field(1).name(), "label");
        assert_eq!(result.schema().field(2).name(), "value"); // Unchanged
    }

    #[test]
    fn test_chain_transform() {
        let batch = create_test_batch();
        let chain = Chain::new()
            .then(Select::new(vec!["id", "value"]))
            .then(Shuffle::with_seed(42));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());

        let result = chain.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
    }

    #[test]
    fn test_empty_chain() {
        let batch = create_test_batch();
        let chain = Chain::new();

        assert!(chain.is_empty());

        let result = chain.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_shuffle_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Shuffle::new();
        let result = transform.apply(batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_select_columns_getter() {
        let select = Select::new(vec!["a", "b"]);
        assert_eq!(select.columns(), &["a", "b"]);
    }

    #[test]
    fn test_shuffle_default() {
        let shuffle = Shuffle::default();
        let batch = create_test_batch();
        let result = shuffle.apply(batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_drop_transform() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["name"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "value");
    }

    #[test]
    fn test_drop_multiple_columns() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["id", "name"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "value");
    }

    #[test]
    fn test_drop_all_columns_error() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["id", "name", "value"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_nonexistent_column_is_ok() {
        let batch = create_test_batch();
        let transform = Drop::new(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 3); // All columns remain
    }

    #[test]
    fn test_drop_columns_getter() {
        let transform = Drop::new(vec!["a", "b"]);
        assert_eq!(transform.columns(), &["a", "b"]);
    }

    #[test]
    fn test_cast_transform() {
        let batch = create_test_batch();
        let transform = Cast::new(vec![("id", DataType::Int64), ("value", DataType::Float64)]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(result.schema().field(2).data_type(), &DataType::Float64);
    }

    #[test]
    fn test_cast_preserves_values() {
        let batch = create_test_batch();
        let transform = Cast::new(vec![("id", DataType::Float64)]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert_eq!(col.value(0), 1.0);
        assert_eq!(col.value(1), 2.0);
        assert_eq!(col.value(2), 3.0);
    }

    #[test]
    fn test_cast_mappings_getter() {
        let transform = Cast::new(vec![("a", DataType::Int64)]);
        assert_eq!(transform.mappings().len(), 1);
        assert_eq!(transform.mappings()[0].0, "a");
        assert_eq!(transform.mappings()[0].1, DataType::Int64);
    }

    #[test]
    fn test_normalize_minmax() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let values = arrow::array::Float64Array::from(vec![0.0, 50.0, 100.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(0) - 0.0).abs() < 1e-10);
        assert!((col.value(1) - 0.5).abs() < 1e-10);
        assert!((col.value(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zscore() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        // Values with mean=0, std=1
        let values = arrow::array::Float64Array::from(vec![-1.0, 0.0, 1.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::ZScore);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // Mean should be 0 (within floating point tolerance)
        let mean: f64 = (0..col.len()).map(|i| col.value(i)).sum::<f64>() / col.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_normalize_l2() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let values = arrow::array::Float64Array::from(vec![3.0, 4.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::L2);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // L2 norm of [3, 4] is 5, so normalized should be [0.6, 0.8]
        assert!((col.value(0) - 0.6).abs() < 1e-10);
        assert!((col.value(1) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_all_numeric() {
        let batch = create_test_batch();
        let transform = Normalize::all_numeric(NormMethod::MinMax);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        // id and value columns should be Float64 now (normalized)
        assert_eq!(result.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(result.schema().field(1).data_type(), &DataType::Utf8); // name unchanged
        assert_eq!(result.schema().field(2).data_type(), &DataType::Float64);
    }

    #[test]
    fn test_normalize_non_numeric_error() {
        let batch = create_test_batch();
        let transform = Normalize::new(vec!["name"], NormMethod::MinMax);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_getters() {
        let transform = Normalize::new(vec!["a", "b"], NormMethod::ZScore);
        assert!(transform.columns().is_some());
        assert_eq!(
            transform
                .columns()
                .unwrap_or_else(|| panic!("Should have columns")),
            &["a", "b"]
        );
        assert_eq!(transform.method(), NormMethod::ZScore);

        let transform2 = Normalize::all_numeric(NormMethod::L2);
        assert!(transform2.columns().is_none());
        assert_eq!(transform2.method(), NormMethod::L2);
    }

    #[test]
    fn test_normalize_constant_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        // All same values - should result in 0 (no range)
        let values = arrow::array::Float64Array::from(vec![5.0, 5.0, 5.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // With constant values, minmax returns 0
        for i in 0..col.len() {
            assert_eq!(col.value(i), 0.0);
        }
    }

    #[test]
    fn test_normalize_with_nulls() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let values = arrow::array::Float64Array::from(vec![Some(0.0), None, Some(100.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(0) - 0.0).abs() < 1e-10);
        assert!(col.is_null(1));
        assert!((col.value(2) - 1.0).abs() < 1e-10);
    }

    // Sample transform tests

    #[test]
    fn test_sample_by_count() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_sample_by_fraction() {
        let batch = create_test_batch();
        let transform = Sample::fraction(0.4).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 2); // 5 * 0.4 = 2
    }

    #[test]
    fn test_sample_deterministic() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result1 = transform.apply(batch.clone());
        let result2 = transform.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let col2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..col1.len() {
            assert_eq!(col1.value(i), col2.value(i));
        }
    }

    #[test]
    fn test_sample_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Check that id-value relationships are preserved
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }
    }

    #[test]
    fn test_sample_count_larger_than_batch() {
        let batch = create_test_batch();
        let transform = Sample::new(100);

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_sample_getters() {
        let sample = Sample::new(10).with_seed(42);
        assert_eq!(sample.count(), Some(10));
        assert!(sample.sample_fraction().is_none());

        let sample2 = Sample::fraction(0.5);
        assert!(sample2.count().is_none());
        assert_eq!(sample2.sample_fraction(), Some(0.5));
    }

    // Take transform tests

    #[test]
    fn test_take_transform() {
        let batch = create_test_batch();
        let transform = Take::new(3);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 2);
        assert_eq!(ids.value(2), 3);
    }

    #[test]
    fn test_take_more_than_available() {
        let batch = create_test_batch();
        let transform = Take::new(100);

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_take_count_getter() {
        let take = Take::new(42);
        assert_eq!(take.count(), 42);
    }

    // Skip transform tests

    #[test]
    fn test_skip_transform() {
        let batch = create_test_batch();
        let transform = Skip::new(2);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(ids.value(0), 3);
        assert_eq!(ids.value(1), 4);
        assert_eq!(ids.value(2), 5);
    }

    #[test]
    fn test_skip_all_rows() {
        let batch = create_test_batch();
        let transform = Skip::new(10);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_skip_count_getter() {
        let skip = Skip::new(5);
        assert_eq!(skip.count(), 5);
    }

    // Sort transform tests

    #[test]
    fn test_sort_ascending() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let values = Int32Array::from(vec![3, 1, 4, 1, 5]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("value");
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 1);
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 4);
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_sort_descending() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let values = Int32Array::from(vec![3, 1, 4, 1, 5]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("value").order(SortOrder::Descending);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(col.value(0), 5);
        assert_eq!(col.value(1), 4);
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 1);
        assert_eq!(col.value(4), 1);
    }

    #[test]
    fn test_sort_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Sort::by("id").order(SortOrder::Descending);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Verify rows are still correlated
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }

        // Verify descending order
        assert_eq!(ids.value(0), 5);
        assert_eq!(ids.value(4), 1);
    }

    #[test]
    fn test_sort_multiple_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let groups = Int32Array::from(vec![1, 2, 1, 2, 1]);
        let values = Int32Array::from(vec![30, 10, 10, 20, 20]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(groups), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by_columns(vec![
            ("group", SortOrder::Ascending),
            ("value", SortOrder::Ascending),
        ]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let groups = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Group 1 first, sorted by value
        assert_eq!(groups.value(0), 1);
        assert_eq!(values.value(0), 10);
        assert_eq!(groups.value(1), 1);
        assert_eq!(values.value(1), 20);
        assert_eq!(groups.value(2), 1);
        assert_eq!(values.value(2), 30);
        // Then group 2
        assert_eq!(groups.value(3), 2);
        assert_eq!(values.value(3), 10);
        assert_eq!(groups.value(4), 2);
        assert_eq!(values.value(4), 20);
    }

    #[test]
    fn test_sort_column_not_found() {
        let batch = create_test_batch();
        let transform = Sort::by("nonexistent");

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_columns_getter() {
        let sort = Sort::by("value").order(SortOrder::Descending);
        let cols = sort.columns();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].0, "value");
        assert_eq!(cols[0].1, SortOrder::Descending);
    }

    #[test]
    fn test_sort_order_default() {
        let order = SortOrder::default();
        assert_eq!(order, SortOrder::Ascending);
    }

    #[test]
    fn test_sort_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("id");
        let result = transform.apply(batch);
        assert!(result.is_ok());
    }

    // FillNull tests

    #[test]
    fn test_fillnull_with_zero_i32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::with_zero("value");
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 0); // Was null, now 0
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 0); // Was null, now 0
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_fillnull_with_int_value() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, Some(3)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Int(-1));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(1), -1);
    }

    #[test]
    fn test_fillnull_with_float() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let values = arrow::array::Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Float(99.9));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(1) - 99.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fillnull_with_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
        let values = StringArray::from(vec![Some("a"), None, Some("c")]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("name", FillStrategy::String("unknown".to_string()));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("Should be StringArray"));

        assert_eq!(col.value(1), "unknown");
    }

    #[test]
    fn test_fillnull_forward_fill() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, None, Some(4), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Forward);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 1); // Forward filled from index 0
        assert_eq!(col.value(2), 1); // Forward filled from index 0
        assert_eq!(col.value(3), 4);
        assert_eq!(col.value(4), 4); // Forward filled from index 3
    }

    #[test]
    fn test_fillnull_backward_fill() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![None, None, Some(3), None, Some(5)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Backward);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 3); // Backward filled from index 2
        assert_eq!(col.value(1), 3); // Backward filled from index 2
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 5); // Backward filled from index 4
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_fillnull_column_not_found() {
        let batch = create_test_batch();
        let transform = FillNull::with_zero("nonexistent");

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_fillnull_getters() {
        let transform = FillNull::new("col", FillStrategy::Int(42));
        assert_eq!(transform.column(), "col");
        assert!(matches!(transform.strategy(), FillStrategy::Int(42)));
    }

    // Unique tests

    #[test]
    fn test_unique_all_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 1]); // Duplicates
        let values = Int32Array::from(vec![10, 20, 10, 20, 30]); // Row 0 == Row 2, Row 1 == Row 3
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::all();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3); // Only 3 unique rows
    }

    #[test]
    fn test_unique_by_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 3]);
        let values = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::by(vec!["id"]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3); // ids 1, 2, 3

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Keep first occurrences by default
        assert_eq!(ids.value(0), 1);
        assert_eq!(values.value(0), 10); // First occurrence of id=1
        assert_eq!(ids.value(1), 2);
        assert_eq!(values.value(1), 20); // First occurrence of id=2
    }

    #[test]
    fn test_unique_keep_last() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 3]);
        let values = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::by(vec!["id"]).keep_last();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Keep last occurrences
        assert_eq!(ids.value(0), 1);
        assert_eq!(values.value(0), 30); // Last occurrence of id=1
        assert_eq!(ids.value(1), 2);
        assert_eq!(values.value(1), 40); // Last occurrence of id=2
    }

    #[test]
    fn test_unique_no_duplicates() {
        let batch = create_test_batch();
        let transform = Unique::all();

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), batch.num_rows()); // All unique
    }

    #[test]
    fn test_unique_column_not_found() {
        let batch = create_test_batch();
        let transform = Unique::by(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_unique_columns_getter() {
        let unique = Unique::by(vec!["a", "b"]);
        assert!(unique.columns().is_some());
        assert_eq!(
            unique
                .columns()
                .unwrap_or_else(|| panic!("Should have columns")),
            &["a", "b"]
        );

        let unique2 = Unique::all();
        assert!(unique2.columns().is_none());
    }

    #[test]
    fn test_unique_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::all();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 1);
    }

    #[test]
    fn test_rename_multiple_columns() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("id", "identifier"), ("name", "label")]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert!(result.schema().field_with_name("identifier").is_ok());
        assert!(result.schema().field_with_name("label").is_ok());
    }

    #[test]
    fn test_rename_nonexistent_column_is_ok() {
        let batch = create_test_batch();
        let transform = Rename::from_pairs([("nonexistent", "new_name")]);
        let result = transform.apply(batch.clone());
        // Renaming a nonexistent column should succeed (no-op)
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_chain_with_multiple_transforms() {
        let batch = create_test_batch();

        let chain = Chain::new()
            .then(Select::new(vec!["id", "name"]))
            .then(Rename::from_pairs([("id", "identifier")]));

        let result = chain.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert!(result.schema().field_with_name("identifier").is_ok());
    }

    #[test]
    fn test_filter_empty_result() {
        let batch = create_test_batch();
        let filter = Filter::new(|batch| Ok(BooleanArray::from(vec![false; batch.num_rows()])));

        let result = filter.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_select_debug() {
        let select = Select::new(vec!["id", "name"]);
        let debug_str = format!("{:?}", select);
        assert!(debug_str.contains("Select"));
    }

    #[test]
    fn test_rename_debug() {
        let rename = Rename::from_pairs([("old", "new")]);
        let debug_str = format!("{:?}", rename);
        assert!(debug_str.contains("Rename"));
    }

    #[test]
    fn test_drop_debug() {
        let drop_t = Drop::new(vec!["col"]);
        let debug_str = format!("{:?}", drop_t);
        assert!(debug_str.contains("Drop"));
    }

    #[test]
    fn test_shuffle_debug() {
        let shuffle = Shuffle::new();
        let debug_str = format!("{:?}", shuffle);
        assert!(debug_str.contains("Shuffle"));
    }

    #[test]
    fn test_cast_debug() {
        let cast = Cast::new(vec![("col", DataType::Int64)]);
        let debug_str = format!("{:?}", cast);
        assert!(debug_str.contains("Cast"));
    }

    #[test]
    fn test_normalize_debug() {
        let normalize = Normalize::new(vec!["col"], NormMethod::MinMax);
        let debug_str = format!("{:?}", normalize);
        assert!(debug_str.contains("Normalize"));
    }

    #[test]
    fn test_fillnull_debug() {
        let fillnull = FillNull::new("col", FillStrategy::Zero);
        let debug_str = format!("{:?}", fillnull);
        assert!(debug_str.contains("FillNull"));
    }

    #[test]
    fn test_sample_debug() {
        let sample = Sample::new(10);
        let debug_str = format!("{:?}", sample);
        assert!(debug_str.contains("Sample"));
    }

    #[test]
    fn test_take_debug() {
        let take = Take::new(10);
        let debug_str = format!("{:?}", take);
        assert!(debug_str.contains("Take"));
    }

    #[test]
    fn test_skip_debug() {
        let skip = Skip::new(5);
        let debug_str = format!("{:?}", skip);
        assert!(debug_str.contains("Skip"));
    }

    #[test]
    fn test_sort_debug() {
        let sort = Sort::by("col");
        let debug_str = format!("{:?}", sort);
        assert!(debug_str.contains("Sort"));
    }

    #[test]
    fn test_unique_debug() {
        let unique = Unique::all();
        let debug_str = format!("{:?}", unique);
        assert!(debug_str.contains("Unique"));
    }

    #[test]
    fn test_sort_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let empty_batch = RecordBatch::new_empty(schema);

        let sort = Sort::by("id");
        let result = sort.apply(empty_batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shuffle_with_seed() {
        let batch = create_test_batch();
        let shuffle = Shuffle::with_seed(12345);

        let result1 = shuffle.apply(batch.clone());
        let result2 = shuffle.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let ids2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..ids1.len() {
            assert_eq!(ids1.value(i), ids2.value(i));
        }
    }

    #[test]
    fn test_sample_with_seed() {
        let batch = create_test_batch();
        let sample = Sample::new(3).with_seed(42);

        let result1 = sample.apply(batch.clone());
        let result2 = sample.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result1.num_rows(), result2.num_rows());
    }

    #[test]
    fn test_cast_int_to_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let cast = Cast::new(vec![("id", DataType::Utf8)]);
        let result = cast.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.schema().field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_cast_nonexistent_column() {
        let batch = create_test_batch();
        let cast = Cast::new(vec![("nonexistent", DataType::Int64)]);
        let result = cast.apply(batch.clone());
        // Cast on nonexistent column is a no-op (doesn't error)
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_skip_more_than_batch_size() {
        let batch = create_test_batch();
        let skip = Skip::new(100);
        let result = skip.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_fillnull_with_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64), None, Some(5i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Int(42));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 42);
        assert_eq!(col.value(3), 42);
    }

    #[test]
    fn test_fillnull_with_float32() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));
        let arr = Float32Array::from(vec![Some(1.0f32), None, Some(3.0f32)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Float(2.5));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert!((col.value(1) - 2.5f32).abs() < 0.001);
    }

    #[test]
    fn test_fillnull_with_bool() {
        use arrow::array::BooleanArray;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )]));
        let arr = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("flag", FillStrategy::Bool(true));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert!(col.value(1));
    }

    #[test]
    fn test_fillnull_zero_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0);
    }

    #[test]
    fn test_fillnull_zero_float32() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));
        let arr = Float32Array::from(vec![Some(1.0f32), None, Some(3.0f32)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0.0f32);
    }

    #[test]
    fn test_fillnull_zero_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0f64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0.0f64);
    }

    #[test]
    fn test_fillnull_forward_fill_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, None, Some(4i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Forward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 1);
        assert_eq!(col.value(2), 1);
    }

    #[test]
    fn test_fillnull_backward_fill_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, None, Some(4i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Backward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 4);
        assert_eq!(col.value(2), 4);
    }

    #[test]
    fn test_fillnull_unsupported_type_strategy() {
        use arrow::array::Date32Array;

        // Date32 with Int fill strategy is not supported
        let schema = Arc::new(Schema::new(vec![Field::new(
            "date",
            DataType::Date32,
            true,
        )]));
        let arr = Date32Array::from(vec![Some(1000), None, Some(3000)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("date", FillStrategy::Int(42));
        let result = fillnull.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_unique_with_int64_column() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let id_arr = Int64Array::from(vec![1i64, 1i64, 2i64, 2i64, 3i64]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["id"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_float64_column() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let val_arr = Float64Array::from(vec![1.0f64, 1.0f64, 2.0f64, 2.0f64, 3.0f64]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(val_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["val"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_float32_column() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let val_arr = Float32Array::from(vec![1.0f32, 1.0f32, 2.0f32, 2.0f32, 3.0f32]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(val_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["val"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_bool_column() {
        use arrow::array::BooleanArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Boolean, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let flag_arr = BooleanArray::from(vec![true, true, false, false, true]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(flag_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["flag"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_unique_with_null_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, false),
        ]));

        let id_arr = Int32Array::from(vec![Some(1), None, Some(1), None, Some(2)]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["id"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // 1, NULL, 2
    }

    #[test]
    fn test_map_with_error() {
        let batch = create_test_batch();
        let map = Map::new(|_batch| Err(crate::Error::transform("intentional error")));
        let result = map.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_fillnull_forward_fill_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![Some(1.0f64), None, None, Some(4.0f64), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Forward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 1.0);
        assert_eq!(col.value(2), 1.0);
        assert_eq!(col.value(4), 4.0);
    }

    #[test]
    fn test_fillnull_backward_fill_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![None, Some(2.0f64), None, None, Some(5.0f64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Backward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(0), 2.0);
        assert_eq!(col.value(2), 5.0);
        assert_eq!(col.value(3), 5.0);
    }

    #[test]
    fn test_filter_closure() {
        let batch = create_test_batch();
        // Test with a closure that filters to only rows where id > 2
        let filter = Filter::new(|batch: &RecordBatch| {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>();
            if let Some(arr) = id_col {
                let mask: Vec<bool> = (0..arr.len()).map(|i| arr.value(i) > 2).collect();
                Ok(arrow::array::BooleanArray::from(mask))
            } else {
                Ok(arrow::array::BooleanArray::from(vec![
                    false;
                    batch.num_rows()
                ]))
            }
        });
        let result = filter.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // rows with id 3, 4, 5
    }

    #[test]
    fn test_select_preserves_column_order() {
        let batch = create_test_batch();
        // Select in reverse order
        let select = Select::new(vec!["value", "name", "id"]);
        let result = select.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.schema().field(0).name(), "value");
        assert_eq!(result.schema().field(1).name(), "name");
        assert_eq!(result.schema().field(2).name(), "id");
    }
}
