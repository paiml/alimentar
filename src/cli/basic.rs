//! Basic CLI commands for data conversion and inspection.

use std::path::{Path, PathBuf};

use arrow::util::pretty::print_batches;

use crate::{ArrowDataset, Dataset};

/// Load a dataset from a file path based on extension.
pub(crate) fn load_dataset(path: &PathBuf) -> crate::Result<ArrowDataset> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => ArrowDataset::from_parquet(path),
        "csv" => ArrowDataset::from_csv(path),
        "json" | "jsonl" => ArrowDataset::from_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

/// Save a dataset to a file path based on extension.
pub(crate) fn save_dataset(dataset: &ArrowDataset, path: &PathBuf) -> crate::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => dataset.to_parquet(path),
        "csv" => dataset.to_csv(path),
        "json" | "jsonl" => dataset.to_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

/// Get format name from file extension.
pub(crate) fn get_format(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => "Parquet",
        Some("arrow" | "ipc") => "Arrow IPC",
        Some("csv") => "CSV",
        Some("json" | "jsonl") => "JSON",
        _ => "Unknown",
    }
}

/// Convert between data formats.
pub(crate) fn cmd_convert(input: &PathBuf, output: &PathBuf) -> crate::Result<()> {
    // Load input (supports parquet, csv)
    let dataset = load_dataset(input)?;

    // Save output (supports parquet, csv)
    save_dataset(&dataset, output)?;

    println!(
        "Converted {} -> {} ({} rows)",
        input.display(),
        output.display(),
        dataset.len()
    );

    Ok(())
}

/// Display dataset information.
pub(crate) fn cmd_info(path: &PathBuf) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    println!("File: {}", path.display());
    println!("Format: {}", get_format(path));
    println!("Rows: {}", dataset.len());
    println!("Batches: {}", dataset.num_batches());
    println!("Columns: {}", dataset.schema().fields().len());
    println!("Size: {} bytes", file_size);

    Ok(())
}

/// Display first N rows of a dataset.
pub(crate) fn cmd_head(path: &PathBuf, rows: usize) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    if dataset.is_empty() {
        println!("Dataset is empty");
        return Ok(());
    }

    // Collect rows into batches
    let mut collected = Vec::new();
    let mut count = 0;

    for batch in dataset.iter() {
        let take = (rows - count).min(batch.num_rows());
        if take > 0 {
            collected.push(batch.slice(0, take));
            count += take;
        }
        if count >= rows {
            break;
        }
    }

    if collected.is_empty() {
        println!("No data to display");
        return Ok(());
    }

    // Print using Arrow's pretty printer
    print_batches(&collected).map_err(crate::Error::Arrow)?;

    if count < dataset.len() {
        println!("... showing {} of {} rows", count, dataset.len());
    }

    Ok(())
}

/// Display dataset schema.
pub(crate) fn cmd_schema(path: &PathBuf) -> crate::Result<()> {
    let dataset = load_dataset(path)?;
    let schema = dataset.schema();

    println!("Schema for {}:", path.display());
    println!();

    for (i, field) in schema.fields().iter().enumerate() {
        let nullable = if field.is_nullable() {
            "nullable"
        } else {
            "not null"
        };
        println!(
            "  {}: {} ({}) [{}]",
            i,
            field.name(),
            field.data_type(),
            nullable
        );
    }

    println!();
    println!("Total columns: {}", schema.fields().len());

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_late_init,
    clippy::redundant_pattern_matching
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_parquet(path: &PathBuf, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_cmd_info() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_info(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_head() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_head(&path, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_schema() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 10);

        let result = cmd_schema(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.parquet");
        create_test_parquet(&input, 50);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());

        // Verify output was created and has same data
        let original = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load original"));
        let converted = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load converted"));

        assert_eq!(original.len(), converted.len());
    }

    #[test]
    fn test_load_dataset_unsupported() {
        let path = PathBuf::from("test.xyz");
        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format() {
        assert_eq!(get_format(Path::new("test.parquet")), "Parquet");
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
        assert_eq!(get_format(Path::new("test.csv")), "CSV");
        assert_eq!(get_format(Path::new("test.json")), "JSON");
        assert_eq!(get_format(Path::new("test.unknown")), "Unknown");
    }

    #[test]
    fn test_cmd_head_with_more_rows_than_dataset() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than exist
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_parquet_to_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 25);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_parquet_to_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 15);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_save_dataset_unsupported_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 5);

        let dataset = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));

        let result = save_dataset(&dataset, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format_ipc() {
        assert_eq!(get_format(Path::new("test.ipc")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_jsonl() {
        assert_eq!(get_format(Path::new("test.jsonl")), "JSON");
    }

    #[test]
    fn test_get_format_no_extension() {
        assert_eq!(get_format(Path::new("testfile")), "Unknown");
    }

    #[test]
    fn test_cmd_convert_unsupported_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 10);

        let result = cmd_convert(&input, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_dataset_xyz_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.xyz");

        std::fs::write(&path, "some data")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format_arrow() {
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_unknown() {
        assert_eq!(get_format(Path::new("test.feather")), "Unknown");
        assert_eq!(get_format(Path::new("test.txt")), "Unknown");
    }

    #[test]
    fn test_load_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let csv_path = temp_dir.path().join("data.csv");

        create_test_parquet(&parquet_path, 10);

        // Convert to CSV first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_csv(&csv_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write csv"));

        // Load from CSV
        let loaded = load_dataset(&csv_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let json_path = temp_dir.path().join("data.json");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&json_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write json"));

        // Load from JSON
        let loaded = load_dataset(&json_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_jsonl() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let jsonl_path = temp_dir.path().join("data.jsonl");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first (jsonl is same format)
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&jsonl_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write jsonl"));

        // Load from JSONL
        let loaded = load_dataset(&jsonl_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_save_dataset_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.csv");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.json");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_unknown_extension() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.xyz");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_convert_to_csv_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_to_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_head_more_than_available() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("small.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than available
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_csv_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let csv_path = temp_dir.path().join("test.csv");

        // Create a simple CSV
        std::fs::write(&csv_path, "id,name\n1,foo\n2,bar\n").unwrap();

        let result = load_dataset(&csv_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_json_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let json_path = temp_dir.path().join("test.json");

        // Create a simple JSON Lines file
        std::fs::write(
            &json_path,
            r#"{"id":1,"name":"foo"}
{"id":2,"name":"bar"}"#,
        )
        .unwrap();

        let result = load_dataset(&json_path);
        assert!(result.is_ok());
    }
}
