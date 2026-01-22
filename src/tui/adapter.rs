//! Dataset adapter for TUI viewing
//!
//! Provides uniform access to Arrow datasets for TUI rendering.
//! Supports zero-copy access to RecordBatch data.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};

use super::error::TuiResult;
use super::format::format_array_value;
use crate::dataset::ArrowDataset;
use crate::Dataset;

/// Adapter providing uniform access to Arrow datasets for TUI rendering
///
/// The adapter caches schema and total row count for O(1) access,
/// while providing safe cell access with bounds checking.
///
/// # Example
///
/// ```ignore
/// use alimentar::tui::DatasetAdapter;
/// use alimentar::ArrowDataset;
///
/// let dataset = ArrowDataset::from_parquet("data.parquet")?;
/// let adapter = DatasetAdapter::from_dataset(&dataset)?;
///
/// println!("Rows: {}", adapter.row_count());
/// println!("Columns: {}", adapter.column_count());
///
/// if let Some(value) = adapter.get_cell(0, 0)? {
///     println!("First cell: {}", value);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DatasetAdapter {
    /// Record batches containing the data
    batches: Vec<RecordBatch>,
    /// Cached schema reference
    schema: SchemaRef,
    /// Cached total row count
    total_rows: usize,
    /// Cached column count
    column_count: usize,
    /// Cumulative row offsets for batch lookup
    batch_offsets: Vec<usize>,
}

impl DatasetAdapter {
    /// Create adapter from an `ArrowDataset`
    ///
    /// # Arguments
    /// * `dataset` - The Arrow dataset to adapt
    ///
    /// # Returns
    /// A new adapter, or error if the dataset has no schema
    pub fn from_dataset(dataset: &ArrowDataset) -> TuiResult<Self> {
        let schema = dataset.schema();
        let batches: Vec<_> = dataset.iter().collect();
        Self::from_batches(batches, schema)
    }

    /// Create adapter from record batches and schema
    ///
    /// # Arguments
    /// * `batches` - Vector of record batches
    /// * `schema` - Schema for the data
    ///
    /// # Returns
    /// A new adapter
    pub fn from_batches(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        let total_rows = batches.iter().map(|b| b.num_rows()).sum();
        let column_count = schema.fields().len();

        // Pre-compute batch offsets for O(1) row lookup
        let mut batch_offsets = Vec::with_capacity(batches.len() + 1);
        batch_offsets.push(0);
        let mut offset = 0;
        for batch in &batches {
            offset += batch.num_rows();
            batch_offsets.push(offset);
        }

        Ok(Self {
            batches,
            schema,
            total_rows,
            column_count,
            batch_offsets,
        })
    }

    /// Create an empty adapter
    ///
    /// Useful for testing edge cases with empty datasets.
    pub fn empty() -> Self {
        Self {
            batches: Vec::new(),
            schema: Arc::new(Schema::empty()),
            total_rows: 0,
            column_count: 0,
            batch_offsets: vec![0],
        }
    }

    /// Get the schema reference
    ///
    /// This is an O(1) operation returning a cached Arc.
    #[inline]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the total row count
    ///
    /// This is an O(1) operation returning a cached value.
    #[inline]
    pub fn row_count(&self) -> usize {
        self.total_rows
    }

    /// Get the column count
    ///
    /// This is an O(1) operation returning a cached value.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Check if the dataset is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// Get a cell value as a formatted string
    ///
    /// # Arguments
    /// * `row` - Global row index (across all batches)
    /// * `col` - Column index
    ///
    /// # Returns
    /// * `Ok(Some(value))` - The formatted cell value
    /// * `Ok(None)` - Row or column out of bounds
    /// * `Err(_)` - Formatting error
    pub fn get_cell(&self, row: usize, col: usize) -> TuiResult<Option<String>> {
        // Bounds check
        if row >= self.total_rows {
            return Ok(None);
        }
        if col >= self.column_count {
            return Ok(None);
        }

        // Find the batch and local row
        let Some((batch_idx, local_row)) = self.locate_row(row) else {
            return Ok(None);
        };

        // Get the batch and column
        let Some(batch) = self.batches.get(batch_idx) else {
            return Ok(None);
        };

        let array = batch.column(col);

        // Format the value
        format_array_value(array.as_ref(), local_row)
    }

    /// Get a field name by column index
    ///
    /// # Arguments
    /// * `col` - Column index
    ///
    /// # Returns
    /// The field name, or None if out of bounds
    pub fn field_name(&self, col: usize) -> Option<&str> {
        self.schema.fields().get(col).map(|f| f.name().as_str())
    }

    /// Get a field data type description by column index
    ///
    /// # Arguments
    /// * `col` - Column index
    ///
    /// # Returns
    /// A string description of the data type
    pub fn field_type(&self, col: usize) -> Option<String> {
        self.schema
            .fields()
            .get(col)
            .map(|f| format!("{:?}", f.data_type()))
    }

    /// Check if a field is nullable
    ///
    /// # Arguments
    /// * `col` - Column index
    ///
    /// # Returns
    /// True if nullable, false if not, None if out of bounds
    pub fn field_nullable(&self, col: usize) -> Option<bool> {
        self.schema.fields().get(col).map(|f| f.is_nullable())
    }

    /// Locate a global row index within the batch structure
    ///
    /// Uses binary search on pre-computed offsets for O(log n) lookup.
    ///
    /// # Arguments
    /// * `global_row` - The global row index
    ///
    /// # Returns
    /// Tuple of (batch_index, local_row_index), or None if out of bounds
    fn locate_row(&self, global_row: usize) -> Option<(usize, usize)> {
        if global_row >= self.total_rows {
            return None;
        }

        // Binary search for the batch containing this row
        let batch_idx = match self.batch_offsets.binary_search(&global_row) {
            Ok(idx) => {
                // Exact match on boundary - belongs to this batch
                if idx < self.batches.len() {
                    idx
                } else {
                    // Edge case: pointing to end
                    idx.saturating_sub(1)
                }
            }
            Err(idx) => {
                // Not exact match - row is in batch idx-1
                idx.saturating_sub(1)
            }
        };

        // Calculate local row within batch
        let batch_start = self.batch_offsets.get(batch_idx).copied().unwrap_or(0);
        let local_row = global_row.saturating_sub(batch_start);

        Some((batch_idx, local_row))
    }

    /// Calculate optimal column widths for display
    ///
    /// Samples rows to determine appropriate column widths,
    /// respecting the maximum total width constraint.
    ///
    /// # Arguments
    /// * `max_width` - Maximum total width available
    /// * `sample_rows` - Number of rows to sample for width estimation
    ///
    /// # Returns
    /// Vector of column widths
    pub fn calculate_column_widths(&self, max_width: u16, sample_rows: usize) -> Vec<u16> {
        if self.column_count == 0 {
            return Vec::new();
        }

        // Start with header widths
        let mut widths: Vec<u16> = self
            .schema
            .fields()
            .iter()
            .map(|f| {
                let len = f.name().len().min(50);
                u16::try_from(len).unwrap_or(u16::MAX)
            })
            .collect();

        // Sample rows for content width
        let sample_count = sample_rows.min(self.total_rows);
        for row in 0..sample_count {
            for col in 0..self.column_count {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    let len = value.chars().take(50).count();
                    let len_u16 = u16::try_from(len).unwrap_or(u16::MAX);
                    if let Some(w) = widths.get_mut(col) {
                        *w = (*w).max(len_u16);
                    }
                }
            }
        }

        // Ensure minimum width of 3 for each column
        for w in &mut widths {
            *w = (*w).max(3);
        }

        // Calculate separators and available space
        let num_cols = u16::try_from(self.column_count).unwrap_or(u16::MAX);
        let separators = num_cols.saturating_sub(1);
        let available = max_width.saturating_sub(separators);

        // Scale down if needed
        let total: u16 = widths.iter().sum();
        if total > available && available > 0 {
            // Scale proportionally
            let scale = f64::from(available) / f64::from(total);
            for w in &mut widths {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let scaled = (f64::from(*w) * scale) as u16;
                *w = scaled.max(3); // Minimum 3 chars
            }
        }

        widths
    }

    /// Get all field names as a vector
    pub fn field_names(&self) -> Vec<&str> {
        self.schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]))
    }

    fn create_test_batch(schema: &SchemaRef, start_id: i32, count: usize) -> RecordBatch {
        let ids: Vec<String> = (0..count)
            .map(|i| format!("id_{}", start_id + i as i32))
            .collect();
        let values: Vec<i32> = (0..count).map(|i| (start_id + i as i32) * 10).collect();
        let scores: Vec<f32> = (0..count).map(|i| (i as f32) * 0.1).collect();

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(Int32Array::from(values)),
                Arc::new(Float32Array::from(scores)),
            ],
        )
        .unwrap()
    }

    fn create_test_adapter() -> DatasetAdapter {
        let schema = create_test_schema();
        let batch1 = create_test_batch(&schema, 0, 5);
        let batch2 = create_test_batch(&schema, 5, 5);
        DatasetAdapter::from_batches(vec![batch1, batch2], schema).unwrap()
    }

    #[test]
    fn f001_adapter_row_count() {
        let adapter = create_test_adapter();
        assert_eq!(adapter.row_count(), 10, "FALSIFIED: Expected 10 rows");
    }

    #[test]
    fn f002_adapter_column_count() {
        let adapter = create_test_adapter();
        assert_eq!(adapter.column_count(), 3, "FALSIFIED: Expected 3 columns");
    }

    #[test]
    fn f003_adapter_schema_fields() {
        let adapter = create_test_adapter();
        let names = adapter.field_names();
        assert_eq!(names, vec!["id", "value", "score"]);
    }

    #[test]
    fn f004_adapter_get_cell_valid() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(0, 0).unwrap();
        assert!(cell.is_some(), "FALSIFIED: First cell should exist");
        assert_eq!(cell.unwrap(), "id_0");
    }

    #[test]
    fn f005_adapter_get_cell_cross_batch() {
        let adapter = create_test_adapter();
        // Row 5 is in batch 2 (batch 1 has rows 0-4)
        let cell = adapter.get_cell(5, 0).unwrap();
        assert!(cell.is_some(), "FALSIFIED: Row 5 should exist");
        assert_eq!(cell.unwrap(), "id_5");
    }

    #[test]
    fn f006_adapter_get_cell_row_out_of_bounds() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(100, 0).unwrap();
        assert!(
            cell.is_none(),
            "FALSIFIED: Out of bounds row should return None"
        );
    }

    #[test]
    fn f007_adapter_get_cell_col_out_of_bounds() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(0, 100).unwrap();
        assert!(
            cell.is_none(),
            "FALSIFIED: Out of bounds column should return None"
        );
    }

    #[test]
    fn f008_adapter_empty() {
        let adapter = DatasetAdapter::empty();
        assert_eq!(adapter.row_count(), 0);
        assert_eq!(adapter.column_count(), 0);
        assert!(adapter.is_empty());
    }

    #[test]
    fn f009_adapter_empty_get_cell() {
        let adapter = DatasetAdapter::empty();
        let cell = adapter.get_cell(0, 0).unwrap();
        assert!(
            cell.is_none(),
            "FALSIFIED: Empty adapter should return None"
        );
    }

    #[test]
    fn f010_adapter_field_name() {
        let adapter = create_test_adapter();
        assert_eq!(adapter.field_name(0), Some("id"));
        assert_eq!(adapter.field_name(1), Some("value"));
        assert_eq!(adapter.field_name(100), None);
    }

    #[test]
    fn f011_adapter_field_type() {
        let adapter = create_test_adapter();
        let type_str = adapter.field_type(0).unwrap();
        assert!(type_str.contains("Utf8"), "FALSIFIED: id should be Utf8");
    }

    #[test]
    fn f012_adapter_field_nullable() {
        let adapter = create_test_adapter();
        assert_eq!(
            adapter.field_nullable(0),
            Some(false),
            "FALSIFIED: id should not be nullable"
        );
    }

    #[test]
    fn f013_adapter_column_widths() {
        let adapter = create_test_adapter();
        let widths = adapter.calculate_column_widths(80, 5);
        assert_eq!(
            widths.len(),
            3,
            "FALSIFIED: Should have width for each column"
        );
        for (i, w) in widths.iter().enumerate() {
            assert!(*w >= 3, "FALSIFIED: Column {} width {} below minimum", i, w);
        }
    }

    #[test]
    fn f014_adapter_column_widths_constrained() {
        let adapter = create_test_adapter();
        // Very tight constraint
        let widths = adapter.calculate_column_widths(15, 5);
        let total: u16 = widths.iter().sum();
        let separators = (widths.len() as u16).saturating_sub(1);
        assert!(
            total + separators <= 15,
            "FALSIFIED: Total width {} exceeds constraint 15",
            total + separators
        );
    }

    #[test]
    fn f015_adapter_locate_row_first_batch() {
        let adapter = create_test_adapter();
        let loc = adapter.locate_row(0);
        assert_eq!(loc, Some((0, 0)), "FALSIFIED: Row 0 should be in batch 0");
    }

    #[test]
    fn f016_adapter_locate_row_second_batch() {
        let adapter = create_test_adapter();
        let loc = adapter.locate_row(5);
        assert_eq!(
            loc,
            Some((1, 0)),
            "FALSIFIED: Row 5 should be first row of batch 1"
        );
    }

    #[test]
    fn f017_adapter_locate_row_last() {
        let adapter = create_test_adapter();
        let loc = adapter.locate_row(9);
        assert_eq!(
            loc,
            Some((1, 4)),
            "FALSIFIED: Row 9 should be last row of batch 1"
        );
    }

    #[test]
    fn f018_adapter_locate_row_out_of_bounds() {
        let adapter = create_test_adapter();
        let loc = adapter.locate_row(100);
        assert_eq!(loc, None, "FALSIFIED: Out of bounds should return None");
    }

    #[test]
    fn f019_adapter_is_clone() {
        let adapter = create_test_adapter();
        let cloned = adapter.clone();
        assert_eq!(adapter.row_count(), cloned.row_count());
        assert_eq!(adapter.column_count(), cloned.column_count());
    }

    #[test]
    fn f020_adapter_schema_o1() {
        let adapter = create_test_adapter();
        // Access schema many times - should be O(1)
        for _ in 0..10000 {
            let _ = adapter.schema();
        }
        // If we get here without timeout, it's O(1)
    }

    #[test]
    fn f021_adapter_row_count_o1() {
        let adapter = create_test_adapter();
        for _ in 0..10000 {
            let _ = adapter.row_count();
        }
        // If we get here without timeout, it's O(1)
    }

    #[test]
    fn f022_adapter_int_formatting() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(0, 1).unwrap().unwrap();
        assert_eq!(cell, "0", "FALSIFIED: First value should be 0");
    }

    #[test]
    fn f023_adapter_float_formatting() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(1, 2).unwrap().unwrap();
        // Float32 formatted to 2 decimal places
        assert!(cell.contains("0.1"), "FALSIFIED: Score should be ~0.1");
    }

    #[test]
    fn f024_adapter_large_row_index() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(usize::MAX, 0).unwrap();
        assert!(cell.is_none(), "FALSIFIED: usize::MAX should not panic");
    }

    #[test]
    fn f025_adapter_large_col_index() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(0, usize::MAX).unwrap();
        assert!(
            cell.is_none(),
            "FALSIFIED: usize::MAX column should not panic"
        );
    }
}
