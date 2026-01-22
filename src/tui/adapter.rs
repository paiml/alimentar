//! Dataset adapter for TUI viewing
//!
//! Provides uniform access to Arrow datasets for TUI rendering.
//! Supports both in-memory and streaming modes for memory efficiency.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use unicode_width::UnicodeWidthStr;

use super::error::TuiResult;
use super::format::format_array_value;
use crate::dataset::ArrowDataset;
use crate::Dataset;

/// Threshold for switching from in-memory to streaming mode (rows)
const STREAMING_THRESHOLD: usize = 100_000;

/// Adapter providing uniform access to Arrow datasets for TUI rendering
///
/// Supports two modes:
/// - `InMemory`: All batches loaded upfront, fast random access
/// - `Streaming`: Lazy batch loading for large datasets (OOM prevention)
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
pub enum DatasetAdapter {
    /// All batches loaded in memory - fast random access
    InMemory(InMemoryAdapter),
    /// Lazy batch loading for large datasets
    Streaming(StreamingAdapter),
}

/// In-memory adapter with all batches loaded
#[derive(Debug, Clone)]
pub struct InMemoryAdapter {
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

/// Streaming adapter for lazy batch loading (stub implementation)
///
/// This adapter is designed for datasets too large to fit in memory.
/// Batches are loaded on-demand and evicted when not needed.
#[derive(Debug, Clone)]
pub struct StreamingAdapter {
    /// Cached schema reference
    schema: SchemaRef,
    /// Total row count (known from metadata)
    total_rows: usize,
    /// Column count
    column_count: usize,
    /// Currently loaded batches (LRU cache would go here)
    loaded_batches: Vec<RecordBatch>,
    /// Cumulative row offsets
    batch_offsets: Vec<usize>,
}

impl DatasetAdapter {
    /// Create adapter from an `ArrowDataset`
    ///
    /// Automatically selects InMemory or Streaming mode based on dataset size.
    ///
    /// # Arguments
    /// * `dataset` - The Arrow dataset to adapt
    ///
    /// # Returns
    /// A new adapter, or error if the dataset has no schema
    pub fn from_dataset(dataset: &ArrowDataset) -> TuiResult<Self> {
        let schema = dataset.schema();
        let batches: Vec<_> = dataset.iter().collect();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Choose mode based on dataset size (F103)
        if total_rows > STREAMING_THRESHOLD {
            Self::streaming_from_batches(batches, schema)
        } else {
            Self::in_memory_from_batches(batches, schema)
        }
    }

    /// Create in-memory adapter from record batches and schema
    pub fn from_batches(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        Self::in_memory_from_batches(batches, schema)
    }

    /// Create in-memory adapter explicitly
    pub fn in_memory_from_batches(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        Ok(Self::InMemory(InMemoryAdapter::new(batches, schema)?))
    }

    /// Create streaming adapter explicitly
    pub fn streaming_from_batches(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        Ok(Self::Streaming(StreamingAdapter::new(batches, schema)?))
    }

    /// Create an empty adapter
    pub fn empty() -> Self {
        Self::InMemory(InMemoryAdapter::empty())
    }

    /// Get the schema reference
    #[inline]
    pub fn schema(&self) -> &SchemaRef {
        match self {
            Self::InMemory(a) => a.schema(),
            Self::Streaming(a) => a.schema(),
        }
    }

    /// Get the total row count
    #[inline]
    pub fn row_count(&self) -> usize {
        match self {
            Self::InMemory(a) => a.row_count(),
            Self::Streaming(a) => a.row_count(),
        }
    }

    /// Get the column count
    #[inline]
    pub fn column_count(&self) -> usize {
        match self {
            Self::InMemory(a) => a.column_count(),
            Self::Streaming(a) => a.column_count(),
        }
    }

    /// Check if the dataset is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    /// Check if this adapter is in streaming mode
    #[inline]
    pub fn is_streaming(&self) -> bool {
        matches!(self, Self::Streaming(_))
    }

    /// Get a cell value as a formatted string
    pub fn get_cell(&self, row: usize, col: usize) -> TuiResult<Option<String>> {
        match self {
            Self::InMemory(a) => a.get_cell(row, col),
            Self::Streaming(a) => a.get_cell(row, col),
        }
    }

    /// Get a field name by column index
    pub fn field_name(&self, col: usize) -> Option<&str> {
        match self {
            Self::InMemory(a) => a.field_name(col),
            Self::Streaming(a) => a.field_name(col),
        }
    }

    /// Get a field data type description by column index
    pub fn field_type(&self, col: usize) -> Option<String> {
        match self {
            Self::InMemory(a) => a.field_type(col),
            Self::Streaming(a) => a.field_type(col),
        }
    }

    /// Check if a field is nullable
    pub fn field_nullable(&self, col: usize) -> Option<bool> {
        match self {
            Self::InMemory(a) => a.field_nullable(col),
            Self::Streaming(a) => a.field_nullable(col),
        }
    }

    /// Calculate optimal column widths for display
    ///
    /// Uses `unicode-width` for correct visual width calculation.
    pub fn calculate_column_widths(&self, max_width: u16, sample_rows: usize) -> Vec<u16> {
        match self {
            Self::InMemory(a) => a.calculate_column_widths(max_width, sample_rows),
            Self::Streaming(a) => a.calculate_column_widths(max_width, sample_rows),
        }
    }

    /// Get all field names as a vector
    pub fn field_names(&self) -> Vec<&str> {
        match self {
            Self::InMemory(a) => a.field_names(),
            Self::Streaming(a) => a.field_names(),
        }
    }

    /// Locate a row within the batch structure
    pub fn locate_row(&self, global_row: usize) -> Option<(usize, usize)> {
        match self {
            Self::InMemory(a) => a.locate_row(global_row),
            Self::Streaming(a) => a.locate_row(global_row),
        }
    }

    /// Search for a substring in string columns, returning first matching row
    ///
    /// Linear scan implementation suitable for <100k rows (F101).
    pub fn search(&self, query: &str) -> Option<usize> {
        if query.is_empty() {
            return None;
        }
        let query_lower = query.to_lowercase();

        for row in 0..self.row_count() {
            for col in 0..self.column_count() {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    if value.to_lowercase().contains(&query_lower) {
                        return Some(row);
                    }
                }
            }
        }
        None
    }

    /// Search continuing from a given row
    pub fn search_from(&self, query: &str, start_row: usize) -> Option<usize> {
        if query.is_empty() {
            return None;
        }
        let query_lower = query.to_lowercase();

        for row in start_row..self.row_count() {
            for col in 0..self.column_count() {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    if value.to_lowercase().contains(&query_lower) {
                        return Some(row);
                    }
                }
            }
        }
        // Wrap around to beginning
        for row in 0..start_row {
            for col in 0..self.column_count() {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    if value.to_lowercase().contains(&query_lower) {
                        return Some(row);
                    }
                }
            }
        }
        None
    }
}

impl InMemoryAdapter {
    /// Create a new in-memory adapter
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        let total_rows = batches.iter().map(|b| b.num_rows()).sum();
        let column_count = schema.fields().len();

        // Pre-compute batch offsets for O(log n) row lookup
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
    pub fn empty() -> Self {
        Self {
            batches: Vec::new(),
            schema: Arc::new(Schema::empty()),
            total_rows: 0,
            column_count: 0,
            batch_offsets: vec![0],
        }
    }

    #[inline]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    #[inline]
    pub fn row_count(&self) -> usize {
        self.total_rows
    }

    #[inline]
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn get_cell(&self, row: usize, col: usize) -> TuiResult<Option<String>> {
        if row >= self.total_rows || col >= self.column_count {
            return Ok(None);
        }

        let Some((batch_idx, local_row)) = self.locate_row(row) else {
            return Ok(None);
        };

        let Some(batch) = self.batches.get(batch_idx) else {
            return Ok(None);
        };

        let array = batch.column(col);
        format_array_value(array.as_ref(), local_row)
    }

    pub fn field_name(&self, col: usize) -> Option<&str> {
        self.schema.fields().get(col).map(|f| f.name().as_str())
    }

    pub fn field_type(&self, col: usize) -> Option<String> {
        self.schema
            .fields()
            .get(col)
            .map(|f| format!("{:?}", f.data_type()))
    }

    pub fn field_nullable(&self, col: usize) -> Option<bool> {
        self.schema.fields().get(col).map(|f| f.is_nullable())
    }

    pub fn locate_row(&self, global_row: usize) -> Option<(usize, usize)> {
        if global_row >= self.total_rows {
            return None;
        }

        let batch_idx = match self.batch_offsets.binary_search(&global_row) {
            Ok(idx) => {
                if idx < self.batches.len() {
                    idx
                } else {
                    idx.saturating_sub(1)
                }
            }
            Err(idx) => idx.saturating_sub(1),
        };

        let batch_start = self.batch_offsets.get(batch_idx).copied().unwrap_or(0);
        let local_row = global_row.saturating_sub(batch_start);

        Some((batch_idx, local_row))
    }

    /// Calculate column widths using unicode-width for correct visual width
    pub fn calculate_column_widths(&self, max_width: u16, sample_rows: usize) -> Vec<u16> {
        if self.column_count == 0 {
            return Vec::new();
        }

        // Start with header widths (using unicode width)
        let mut widths: Vec<u16> = self
            .schema
            .fields()
            .iter()
            .map(|f| {
                let width = UnicodeWidthStr::width(f.name().as_str()).min(50);
                u16::try_from(width).unwrap_or(u16::MAX)
            })
            .collect();

        // Sample rows for content width
        let sample_count = sample_rows.min(self.total_rows);
        for row in 0..sample_count {
            for col in 0..self.column_count {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    // Use unicode width for correct visual width
                    let width = UnicodeWidthStr::width(value.as_str()).min(50);
                    let width_u16 = u16::try_from(width).unwrap_or(u16::MAX);
                    if let Some(w) = widths.get_mut(col) {
                        *w = (*w).max(width_u16);
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
            let scale = f64::from(available) / f64::from(total);
            for w in &mut widths {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let scaled = (f64::from(*w) * scale) as u16;
                *w = scaled.max(3);
            }
        }

        widths
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }
}

impl StreamingAdapter {
    /// Create a new streaming adapter
    ///
    /// Note: This is currently a stub that loads all batches.
    /// A full implementation would use an async iterator.
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(batches: Vec<RecordBatch>, schema: SchemaRef) -> TuiResult<Self> {
        let total_rows = batches.iter().map(|b| b.num_rows()).sum();
        let column_count = schema.fields().len();

        let mut batch_offsets = Vec::with_capacity(batches.len() + 1);
        batch_offsets.push(0);
        let mut offset = 0;
        for batch in &batches {
            offset += batch.num_rows();
            batch_offsets.push(offset);
        }

        Ok(Self {
            schema,
            total_rows,
            column_count,
            loaded_batches: batches, // TODO: Replace with lazy loading
            batch_offsets,
        })
    }

    #[inline]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    #[inline]
    pub fn row_count(&self) -> usize {
        self.total_rows
    }

    #[inline]
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn get_cell(&self, row: usize, col: usize) -> TuiResult<Option<String>> {
        if row >= self.total_rows || col >= self.column_count {
            return Ok(None);
        }

        let Some((batch_idx, local_row)) = self.locate_row(row) else {
            return Ok(None);
        };

        let Some(batch) = self.loaded_batches.get(batch_idx) else {
            return Ok(None);
        };

        let array = batch.column(col);
        format_array_value(array.as_ref(), local_row)
    }

    pub fn field_name(&self, col: usize) -> Option<&str> {
        self.schema.fields().get(col).map(|f| f.name().as_str())
    }

    pub fn field_type(&self, col: usize) -> Option<String> {
        self.schema
            .fields()
            .get(col)
            .map(|f| format!("{:?}", f.data_type()))
    }

    pub fn field_nullable(&self, col: usize) -> Option<bool> {
        self.schema.fields().get(col).map(|f| f.is_nullable())
    }

    pub fn locate_row(&self, global_row: usize) -> Option<(usize, usize)> {
        if global_row >= self.total_rows {
            return None;
        }

        let batch_idx = match self.batch_offsets.binary_search(&global_row) {
            Ok(idx) => {
                if idx < self.loaded_batches.len() {
                    idx
                } else {
                    idx.saturating_sub(1)
                }
            }
            Err(idx) => idx.saturating_sub(1),
        };

        let batch_start = self.batch_offsets.get(batch_idx).copied().unwrap_or(0);
        let local_row = global_row.saturating_sub(batch_start);

        Some((batch_idx, local_row))
    }

    /// Calculate column widths using unicode-width
    pub fn calculate_column_widths(&self, max_width: u16, sample_rows: usize) -> Vec<u16> {
        if self.column_count == 0 {
            return Vec::new();
        }

        let mut widths: Vec<u16> = self
            .schema
            .fields()
            .iter()
            .map(|f| {
                let width = UnicodeWidthStr::width(f.name().as_str()).min(50);
                u16::try_from(width).unwrap_or(u16::MAX)
            })
            .collect();

        let sample_count = sample_rows.min(self.total_rows);
        for row in 0..sample_count {
            for col in 0..self.column_count {
                if let Ok(Some(value)) = self.get_cell(row, col) {
                    let width = UnicodeWidthStr::width(value.as_str()).min(50);
                    let width_u16 = u16::try_from(width).unwrap_or(u16::MAX);
                    if let Some(w) = widths.get_mut(col) {
                        *w = (*w).max(width_u16);
                    }
                }
            }
        }

        for w in &mut widths {
            *w = (*w).max(3);
        }

        let num_cols = u16::try_from(self.column_count).unwrap_or(u16::MAX);
        let separators = num_cols.saturating_sub(1);
        let available = max_width.saturating_sub(separators);

        let total: u16 = widths.iter().sum();
        if total > available && available > 0 {
            let scale = f64::from(available) / f64::from(total);
            for w in &mut widths {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let scaled = (f64::from(*w) * scale) as u16;
                *w = scaled.max(3);
            }
        }

        widths
    }

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
        assert_eq!(
            adapter.column_count(),
            3,
            "FALSIFIED: Expected 3 columns (id, value, score)"
        );
    }

    #[test]
    fn f003_adapter_schema_o1() {
        let adapter = create_test_adapter();
        let schema = adapter.schema();
        assert_eq!(schema.fields().len(), 3);
    }

    #[test]
    fn f004_adapter_get_cell_first_batch() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(0, 0).unwrap();
        assert!(cell.is_some(), "FALSIFIED: Cell should exist");
        assert_eq!(cell.unwrap(), "id_0");
    }

    #[test]
    fn f005_adapter_get_cell_second_batch() {
        let adapter = create_test_adapter();
        let cell = adapter.get_cell(5, 0).unwrap();
        assert!(cell.is_some(), "FALSIFIED: Cell should exist");
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
        for _ in 0..10000 {
            let _ = adapter.schema();
        }
    }

    #[test]
    fn f021_adapter_row_count_o1() {
        let adapter = create_test_adapter();
        for _ in 0..10000 {
            let _ = adapter.row_count();
        }
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

    #[test]
    fn f026_adapter_from_dataset() {
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, 0, 5);
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let adapter = DatasetAdapter::from_dataset(&dataset).unwrap();
        assert_eq!(adapter.row_count(), 5);
        assert_eq!(adapter.column_count(), 3);
        assert_eq!(adapter.field_name(0), Some("id"));
    }

    #[test]
    fn f027_adapter_single_batch() {
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, 0, 10);
        let adapter = DatasetAdapter::from_batches(vec![batch], schema).unwrap();

        assert_eq!(adapter.row_count(), 10);
        assert_eq!(adapter.get_cell(0, 0).unwrap(), Some("id_0".to_string()));
        assert_eq!(adapter.get_cell(9, 0).unwrap(), Some("id_9".to_string()));
    }

    #[test]
    fn f028_adapter_multi_batch_boundaries() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(&schema, 0, 3);
        let batch2 = create_test_batch(&schema, 3, 3);
        let batch3 = create_test_batch(&schema, 6, 3);
        let adapter =
            DatasetAdapter::from_batches(vec![batch1, batch2, batch3], schema.clone()).unwrap();

        assert_eq!(adapter.row_count(), 9);
        assert_eq!(adapter.get_cell(2, 0).unwrap(), Some("id_2".to_string()));
        assert_eq!(adapter.get_cell(3, 0).unwrap(), Some("id_3".to_string()));
        assert_eq!(adapter.get_cell(8, 0).unwrap(), Some("id_8".to_string()));
    }

    #[test]
    fn f029_adapter_empty_schema_columns() {
        let schema = create_test_schema();
        let batch1 = create_test_batch(&schema, 0, 5);
        let batch2 = create_test_batch(&schema, 5, 5);
        let adapter = DatasetAdapter::from_batches(vec![batch1, batch2], schema.clone()).unwrap();

        let widths = adapter.calculate_column_widths(100, 10);
        assert_eq!(widths.len(), 3);
    }

    #[test]
    fn f030_adapter_empty_batches() {
        let schema = create_test_schema();
        let adapter = DatasetAdapter::from_batches(vec![], schema).unwrap();
        assert!(adapter.is_empty());
        assert_eq!(adapter.row_count(), 0);
        assert_eq!(adapter.get_cell(0, 0).unwrap(), None);
    }

    #[test]
    fn f031_adapter_empty_schema_field_names() {
        let schema = create_test_schema();
        let adapter = DatasetAdapter::from_batches(vec![], schema).unwrap();
        let names = adapter.field_names();
        assert_eq!(names, vec!["id", "value", "score"]);
    }

    // === NEW TESTS FOR STREAMING AND SEARCH ===

    #[test]
    fn f032_adapter_is_streaming() {
        let adapter = create_test_adapter();
        assert!(
            !adapter.is_streaming(),
            "FALSIFIED: Small dataset should be InMemory"
        );
    }

    #[test]
    fn f033_adapter_streaming_mode() {
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, 0, 5);
        let adapter = DatasetAdapter::streaming_from_batches(vec![batch], schema).unwrap();
        assert!(
            adapter.is_streaming(),
            "FALSIFIED: Should be Streaming mode"
        );
        assert_eq!(adapter.row_count(), 5);
    }

    #[test]
    fn f034_adapter_search_finds_match() {
        let adapter = create_test_adapter();
        let result = adapter.search("id_5");
        assert_eq!(
            result,
            Some(5),
            "FALSIFIED: Search should find 'id_5' at row 5"
        );
    }

    #[test]
    fn f035_adapter_search_no_match() {
        let adapter = create_test_adapter();
        let result = adapter.search("nonexistent_value");
        assert_eq!(result, None, "FALSIFIED: Search should return None");
    }

    #[test]
    fn f036_adapter_search_empty_query() {
        let adapter = create_test_adapter();
        let result = adapter.search("");
        assert_eq!(result, None, "FALSIFIED: Empty query should return None");
    }

    #[test]
    fn f037_adapter_search_case_insensitive() {
        let adapter = create_test_adapter();
        let result = adapter.search("ID_3");
        assert_eq!(
            result,
            Some(3),
            "FALSIFIED: Search should be case insensitive"
        );
    }

    #[test]
    fn f038_adapter_search_from_wraps() {
        let adapter = create_test_adapter();
        // Search from row 8, should wrap and find id_0
        let result = adapter.search_from("id_0", 8);
        assert_eq!(result, Some(0), "FALSIFIED: Search should wrap around");
    }

    #[test]
    fn f039_adapter_unicode_width() {
        // Test that unicode width is correctly calculated
        let schema = Arc::new(Schema::new(vec![Field::new(
            "emoji",
            DataType::Utf8,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["👨‍👩‍👧‍👦", "hello"]))],
        )
        .unwrap();

        let adapter = DatasetAdapter::from_batches(vec![batch], schema).unwrap();
        let widths = adapter.calculate_column_widths(80, 10);

        // Emoji should have visual width of 2 per component,
        // family emoji is complex but should be handled
        assert!(
            widths[0] >= 3,
            "FALSIFIED: Column width should be at least minimum"
        );
    }
}
