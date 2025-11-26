//! Streaming dataset for lazy/chunked data loading.
//!
//! Provides [`StreamingDataset`] for working with datasets that are too large
//! to fit in memory, or when you want to start processing before the full
//! dataset is loaded.

use std::{collections::VecDeque, path::Path, sync::Arc};

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::{Error, Result};

/// A data source that can produce RecordBatches on demand.
pub trait DataSource: Send {
    /// Returns the schema of the data.
    fn schema(&self) -> SchemaRef;

    /// Returns the next batch of data, or None if exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error if reading the next batch fails.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    /// Returns an estimate of total rows, if known.
    fn size_hint(&self) -> Option<usize> {
        None
    }

    /// Resets the source to the beginning, if supported.
    ///
    /// # Errors
    ///
    /// Returns an error if this data source does not support reset.
    fn reset(&mut self) -> Result<()> {
        Err(Error::storage("This data source does not support reset"))
    }
}

/// A streaming dataset that loads data lazily in chunks.
///
/// Unlike [`ArrowDataset`](crate::ArrowDataset) which loads all data into
/// memory, `StreamingDataset` fetches data on-demand, making it suitable for:
/// - Large datasets that don't fit in memory
/// - Network-based data sources where you want to start processing early
/// - Infinite or very large data streams
///
/// # Example
///
/// ```no_run
/// use alimentar::streaming::StreamingDataset;
///
/// let dataset = StreamingDataset::from_parquet("large_data.parquet", 1024).unwrap();
///
/// for batch in dataset {
///     println!("Processing batch with {} rows", batch.num_rows());
/// }
/// ```
pub struct StreamingDataset {
    source: Box<dyn DataSource>,
    buffer: VecDeque<RecordBatch>,
    buffer_size: usize,
    prefetch: usize,
    schema: SchemaRef,
    exhausted: bool,
}

impl StreamingDataset {
    /// Creates a new streaming dataset from a data source.
    ///
    /// # Arguments
    ///
    /// * `source` - The data source to stream from
    /// * `buffer_size` - Maximum number of batches to buffer
    pub fn new(source: Box<dyn DataSource>, buffer_size: usize) -> Self {
        let schema = source.schema();
        Self {
            source,
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size: buffer_size.max(1),
            prefetch: 1,
            schema,
            exhausted: false,
        }
    }

    /// Creates a streaming dataset from a Parquet file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of rows per batch
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or is invalid.
    pub fn from_parquet(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let source = ParquetSource::new(path, batch_size)?;
        Ok(Self::new(Box::new(source), 4))
    }

    /// Sets the number of batches to prefetch.
    ///
    /// Higher values reduce latency but increase memory usage.
    #[must_use]
    pub fn prefetch(mut self, count: usize) -> Self {
        self.prefetch = count.max(1);
        self
    }

    /// Returns the schema of the dataset.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Returns an estimate of total rows, if known.
    pub fn size_hint(&self) -> Option<usize> {
        self.source.size_hint()
    }

    /// Fills the buffer up to the prefetch limit.
    fn fill_buffer(&mut self) -> Result<()> {
        while !self.exhausted && self.buffer.len() < self.prefetch {
            if let Some(batch) = self.source.next_batch()? {
                self.buffer.push_back(batch);
            } else {
                self.exhausted = true;
                break;
            }
        }
        Ok(())
    }

    /// Resets the dataset to the beginning, if the source supports it.
    ///
    /// # Errors
    ///
    /// Returns an error if the source does not support reset.
    pub fn reset(&mut self) -> Result<()> {
        self.source.reset()?;
        self.buffer.clear();
        self.exhausted = false;
        Ok(())
    }
}

impl Iterator for StreamingDataset {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to fill buffer first
        if let Err(_e) = self.fill_buffer() {
            return None;
        }

        self.buffer.pop_front()
    }
}

impl std::fmt::Debug for StreamingDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingDataset")
            .field("buffer_size", &self.buffer_size)
            .field("prefetch", &self.prefetch)
            .field("buffered", &self.buffer.len())
            .field("exhausted", &self.exhausted)
            .finish_non_exhaustive()
    }
}

/// A data source that reads from a Parquet file.
pub struct ParquetSource {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    schema: SchemaRef,
    path: std::path::PathBuf,
    batch_size: usize,
}

impl ParquetSource {
    /// Creates a new Parquet source.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn new(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|e| Error::io(e, &path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(Error::Parquet)?
            .with_batch_size(batch_size);

        let schema = builder.schema().clone();
        let reader = builder.build().map_err(Error::Parquet)?;

        Ok(Self {
            reader,
            schema,
            path,
            batch_size,
        })
    }
}

impl DataSource for ParquetSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(Error::Arrow(e)),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<()> {
        // Re-open the file
        let file = std::fs::File::open(&self.path).map_err(|e| Error::io(e, &self.path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(Error::Parquet)?
            .with_batch_size(self.batch_size);

        self.reader = builder.build().map_err(Error::Parquet)?;
        Ok(())
    }
}

impl std::fmt::Debug for ParquetSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetSource")
            .field("path", &self.path)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

/// A data source backed by in-memory RecordBatches.
///
/// Useful for testing or when you have data already in memory
/// but want to use the streaming interface.
#[derive(Debug)]
pub struct MemorySource {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    position: usize,
}

impl MemorySource {
    /// Creates a new memory source from a vector of batches.
    ///
    /// # Errors
    ///
    /// Returns an error if the batches vector is empty.
    pub fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let schema = batches[0].schema();
        Ok(Self {
            batches,
            schema,
            position: 0,
        })
    }
}

impl DataSource for MemorySource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.position >= self.batches.len() {
            return Ok(None);
        }

        let batch = self.batches[self.position].clone();
        self.position += 1;
        Ok(Some(batch))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.batches.iter().map(|b| b.num_rows()).sum())
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
}

/// A data source that chains multiple sources together.
pub struct ChainedSource {
    sources: Vec<Box<dyn DataSource>>,
    current: usize,
    schema: SchemaRef,
}

impl ChainedSource {
    /// Creates a new chained source.
    ///
    /// # Errors
    ///
    /// Returns an error if the sources vector is empty.
    pub fn new(sources: Vec<Box<dyn DataSource>>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::invalid_config(
                "ChainedSource requires at least one source",
            ));
        }

        let schema = sources[0].schema();
        Ok(Self {
            sources,
            current: 0,
            schema,
        })
    }
}

impl DataSource for ChainedSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        while self.current < self.sources.len() {
            match self.sources[self.current].next_batch()? {
                Some(batch) => return Ok(Some(batch)),
                None => self.current += 1,
            }
        }
        Ok(None)
    }

    fn size_hint(&self) -> Option<usize> {
        let mut total = 0;
        for source in &self.sources {
            match source.size_hint() {
                Some(hint) => total += hint,
                None => return None,
            }
        }
        Some(total)
    }

    fn reset(&mut self) -> Result<()> {
        for source in &mut self.sources {
            source.reset()?;
        }
        self.current = 0;
        Ok(())
    }
}

impl std::fmt::Debug for ChainedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedSource")
            .field("num_sources", &self.sources.len())
            .field("current", &self.current)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_batch(start: i32, count: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (start..start + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"))
    }

    #[test]
    fn test_memory_source() {
        let batches = vec![
            create_test_batch(0, 5),
            create_test_batch(5, 5),
            create_test_batch(10, 5),
        ];

        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        assert_eq!(source.size_hint(), Some(15));

        let mut count = 0;
        while let Ok(Some(batch)) = source.next_batch() {
            count += batch.num_rows();
        }
        assert_eq!(count, 15);
    }

    #[test]
    fn test_memory_source_reset() {
        let batches = vec![create_test_batch(0, 5)];

        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Consume
        let _ = source.next_batch();
        assert!(source.next_batch().ok().flatten().is_none());

        // Reset and consume again
        source
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));
        assert!(source.next_batch().ok().flatten().is_some());
    }

    #[test]
    fn test_streaming_dataset_from_memory() {
        let batches = vec![create_test_batch(0, 10), create_test_batch(10, 10)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);

        let mut total = 0;
        for batch in dataset {
            total += batch.num_rows();
        }
        assert_eq!(total, 20);
    }

    #[test]
    fn test_streaming_dataset_prefetch() {
        let batches = vec![
            create_test_batch(0, 5),
            create_test_batch(5, 5),
            create_test_batch(10, 5),
        ];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(2);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_streaming_dataset_schema() {
        let batches = vec![create_test_batch(0, 5)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);

        assert_eq!(dataset.schema().fields().len(), 2);
        assert_eq!(dataset.schema().field(0).name(), "id");
    }

    #[test]
    fn test_streaming_dataset_reset() {
        let batches = vec![create_test_batch(0, 5)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = StreamingDataset::new(Box::new(source), 4);

        // Consume
        let first: Vec<RecordBatch> = dataset.by_ref().collect();
        assert_eq!(first.len(), 1);

        // Reset
        dataset
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        // Consume again
        let second: Vec<RecordBatch> = dataset.collect();
        assert_eq!(second.len(), 1);
    }

    #[test]
    fn test_chained_source() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source2 = MemorySource::new(vec![create_test_batch(5, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        assert_eq!(chained.size_hint(), Some(10));

        let mut total = 0;
        while let Ok(Some(batch)) = chained.next_batch() {
            total += batch.num_rows();
        }
        assert_eq!(total, 10);
    }

    #[test]
    fn test_empty_memory_source_error() {
        let result = MemorySource::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_chained_source_error() {
        let result = ChainedSource::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parquet_source_roundtrip() {
        // Create test data
        let batch = create_test_batch(0, 100);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        // Write to temp file
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        // Read back via streaming
        let streaming = StreamingDataset::from_parquet(&path, 25)
            .ok()
            .unwrap_or_else(|| panic!("Should create streaming"));

        let total: usize = streaming.map(|b| b.num_rows()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_streaming_dataset_debug() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let dataset = StreamingDataset::new(Box::new(source), 4);

        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("StreamingDataset"));
    }

    #[test]
    fn test_parquet_source_debug() {
        // Create test data
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("debug_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("ParquetSource"));
        assert!(debug_str.contains("debug_test.parquet"));
    }

    #[test]
    fn test_chained_source_debug() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let chained = ChainedSource::new(vec![Box::new(source1)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        let debug_str = format!("{:?}", chained);
        assert!(debug_str.contains("ChainedSource"));
        assert!(debug_str.contains("num_sources"));
    }

    #[test]
    fn test_streaming_dataset_size_hint() {
        let batches = vec![create_test_batch(0, 10), create_test_batch(10, 15)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);
        assert_eq!(dataset.size_hint(), Some(25));
    }

    #[test]
    fn test_parquet_source_reset() {
        // Create test data
        let batch = create_test_batch(0, 50);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("reset_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let mut source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Read all batches
        let mut count = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count += 1;
        }
        assert!(count > 0);

        // Reset and read again
        source
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        let mut count2 = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count2 += 1;
        }
        assert_eq!(count, count2);
    }

    #[test]
    fn test_chained_source_reset() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source2 = MemorySource::new(vec![create_test_batch(5, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        // Exhaust
        let mut count = 0;
        while let Ok(Some(_)) = chained.next_batch() {
            count += 1;
        }
        assert_eq!(count, 2);

        // Reset
        chained
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        // Read again
        let mut count2 = 0;
        while let Ok(Some(_)) = chained.next_batch() {
            count2 += 1;
        }
        assert_eq!(count2, 2);
    }

    #[test]
    fn test_chained_source_size_hint_unknown() {
        // Create a source that doesn't know its size
        struct UnknownSizeSource {
            schema: SchemaRef,
            count: usize,
        }

        impl DataSource for UnknownSizeSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                if self.count > 0 {
                    self.count -= 1;
                    Ok(Some(create_test_batch(0, 1)))
                } else {
                    Ok(None)
                }
            }

            // Returns None, so chained source can't calculate total
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let memory_source = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let unknown_source = UnknownSizeSource { schema, count: 1 };

        let chained = ChainedSource::new(vec![Box::new(memory_source), Box::new(unknown_source)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        // One source has unknown size, so total is None
        assert_eq!(chained.size_hint(), None);
    }

    #[test]
    fn test_streaming_dataset_buffer_size_minimum() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Buffer size 0 should be treated as 1
        let dataset = StreamingDataset::new(Box::new(source), 0);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_streaming_dataset_prefetch_minimum() {
        let batches = vec![create_test_batch(0, 5), create_test_batch(5, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Prefetch 0 should be treated as 1
        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(0);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_memory_source_debug() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("MemorySource"));
    }

    #[test]
    fn test_parquet_source_schema() {
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("schema_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let schema = source.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
    }

    #[test]
    fn test_chained_source_schema() {
        let source = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let chained = ChainedSource::new(vec![Box::new(source)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        let schema = chained.schema();
        assert_eq!(schema.fields().len(), 2);
    }

    #[test]
    fn test_parquet_source_file_not_found() {
        let result = ParquetSource::new("/nonexistent/path/to/file.parquet", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_dataset_from_parquet_not_found() {
        let result = StreamingDataset::from_parquet("/nonexistent/file.parquet", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_source_default_reset_error() {
        // Test that the default DataSource::reset returns error
        struct NoResetSource {
            schema: SchemaRef,
        }

        impl DataSource for NoResetSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
            // Don't override reset() - uses default impl that returns error
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let mut source = NoResetSource { schema };
        let result = source.reset();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_dataset_reset_unsupported() {
        // Test that reset fails when source doesn't support it
        struct NoResetSource {
            schema: SchemaRef,
            done: bool,
        }

        impl DataSource for NoResetSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                if self.done {
                    Ok(None)
                } else {
                    self.done = true;
                    Ok(Some(create_test_batch(0, 5)))
                }
            }
            // Uses default reset which fails
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let source = NoResetSource {
            schema,
            done: false,
        };
        let mut dataset = StreamingDataset::new(Box::new(source), 4);

        // Consume
        let _: Vec<_> = dataset.by_ref().collect();

        // Reset should fail
        let result = dataset.reset();
        assert!(result.is_err());
    }
}
