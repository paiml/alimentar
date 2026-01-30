//! Tests for HuggingFace Hub integration.

use std::path::Path;

use super::*;

#[test]
fn test_builder_basic() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "squad");
    assert_eq!(dataset.revision(), "main");
    assert!(dataset.subset().is_none());
    assert!(dataset.split().is_none());
}

#[test]
fn test_builder_with_options() {
    let dataset = HfDataset::builder("glue")
        .revision("v1.0.0")
        .subset("cola")
        .split("validation")
        .cache_dir("/tmp/test_cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "glue");
    assert_eq!(dataset.revision(), "v1.0.0");
    assert_eq!(dataset.subset(), Some("cola"));
    assert_eq!(dataset.split(), Some("validation"));
    assert_eq!(dataset.cache_dir(), Path::new("/tmp/test_cache"));
}

#[test]
fn test_builder_empty_repo_id_error() {
    let result = HfDataset::builder("").build();
    assert!(result.is_err());
}

#[test]
fn test_build_parquet_path_default() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/train.parquet");
}

#[test]
fn test_build_parquet_path_with_subset() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .split("validation")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "cola/validation.parquet");
}

#[test]
fn test_build_download_url() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/train.parquet");
    assert_eq!(
        url,
        "https://huggingface.co/datasets/squad/resolve/main/data/default/train.parquet"
    );
}

#[test]
fn test_cache_path() {
    let dataset = HfDataset::builder("squad")
        .cache_dir("/tmp/cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("default/train.parquet");
    assert_eq!(
        cache_path,
        std::path::PathBuf::from(
            "/tmp/cache/huggingface/datasets/squad/main/default/train.parquet"
        )
    );
}

#[test]
fn test_default_cache_dir() {
    let cache = default_cache_dir();
    // Should return some path
    assert!(!cache.as_os_str().is_empty());
}

#[test]
fn test_namespaced_repo_id() {
    let dataset = HfDataset::builder("openai/gsm8k")
        .split("test")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/test.parquet");
    assert!(url.contains("openai/gsm8k"));
}

#[test]
fn test_builder_clone() {
    let builder = HfDatasetBuilder::new("squad")
        .revision("v1.0")
        .subset("test")
        .split("validation");

    let cloned = builder;
    let dataset = cloned
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "squad");
    assert_eq!(dataset.revision(), "v1.0");
    assert_eq!(dataset.subset(), Some("test"));
    assert_eq!(dataset.split(), Some("validation"));
}

#[test]
fn test_hf_dataset_clone() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cloned = dataset.clone();
    assert_eq!(cloned.repo_id(), dataset.repo_id());
    assert_eq!(cloned.revision(), dataset.revision());
    assert_eq!(cloned.subset(), dataset.subset());
}

#[test]
fn test_hf_dataset_debug() {
    let dataset = HfDataset::builder("test-dataset")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let debug_str = format!("{:?}", dataset);
    assert!(debug_str.contains("HfDataset"));
    assert!(debug_str.contains("test-dataset"));
}

#[test]
fn test_builder_debug() {
    let builder = HfDatasetBuilder::new("debug-test");
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("HfDatasetBuilder"));
    assert!(debug_str.contains("debug-test"));
}

#[test]
fn test_dataset_info_debug() {
    let info = DatasetInfo {
        repo_id: "test".to_string(),
        splits: vec!["train".to_string(), "test".to_string()],
        subsets: vec!["default".to_string()],
        download_size: Some(1024),
        description: Some("A test dataset".to_string()),
    };

    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("DatasetInfo"));
    assert!(debug_str.contains("test"));
}

#[test]
fn test_dataset_info_clone() {
    let info = DatasetInfo {
        repo_id: "clone-test".to_string(),
        splits: vec!["train".to_string()],
        subsets: vec![],
        download_size: None,
        description: None,
    };

    let cloned = info.clone();
    assert_eq!(cloned.repo_id, info.repo_id);
    assert_eq!(cloned.splits, info.splits);
}

#[test]
fn test_build_parquet_path_train_split() {
    let dataset = HfDataset::builder("squad")
        .split("train")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/train.parquet");
}

#[test]
fn test_build_parquet_path_test_split() {
    let dataset = HfDataset::builder("squad")
        .split("test")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/test.parquet");
}

#[test]
fn test_build_download_url_with_revision() {
    let dataset = HfDataset::builder("squad")
        .revision("refs/convert/parquet")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/train.parquet");
    assert!(url.contains("refs/convert/parquet"));
}

#[test]
fn test_cache_path_with_subset() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .split("validation")
        .cache_dir("/tmp/hf-cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("cola/validation.parquet");
    assert!(cache_path
        .to_string_lossy()
        .contains("/tmp/hf-cache/huggingface/datasets/glue/main/cola/validation.parquet"));
}

#[test]
fn test_clear_cache_nonexistent() {
    let dataset = HfDataset::builder("nonexistent-dataset")
        .cache_dir("/tmp/nonexistent-cache-dir-12345")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Should not error on non-existent cache
    let result = dataset.clear_cache();
    assert!(result.is_ok());
}

#[test]
fn test_download_from_cache() {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use crate::Dataset;

    // Create temp dir for cache
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    // Create HfDataset pointing to this cache
    let dataset = HfDataset::builder("test-repo")
        .cache_dir(temp_dir.path())
        .split("train")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Create the cache directory structure
    let cache_path = dataset.cache_path_for(&dataset.build_parquet_path());
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)
            .ok()
            .unwrap_or_else(|| panic!("Should create dirs"));
    }

    // Create a minimal parquet file in cache
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

    let arrow_dataset = crate::ArrowDataset::from_batch(batch)
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"));

    arrow_dataset
        .to_parquet(&cache_path)
        .ok()
        .unwrap_or_else(|| panic!("Should write parquet"));

    // Now download should use cache
    let loaded = dataset.download();
    assert!(loaded.is_ok());
    let loaded = loaded.ok().unwrap_or_else(|| panic!("Should load"));
    assert_eq!(loaded.len(), 3);
}

#[test]
fn test_clear_cache_with_files() {
    // Create temp dir
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let dataset = HfDataset::builder("clear-test")
        .cache_dir(temp_dir.path())
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Create cache directory with a file
    let cache_dir = temp_dir
        .path()
        .join("huggingface")
        .join("datasets")
        .join("clear-test");
    std::fs::create_dir_all(&cache_dir)
        .ok()
        .unwrap_or_else(|| panic!("Should create dir"));
    std::fs::write(cache_dir.join("test.txt"), "test data")
        .ok()
        .unwrap_or_else(|| panic!("Should write file"));

    // Verify it exists
    assert!(cache_dir.exists());

    // Clear cache
    let result = dataset.clear_cache();
    assert!(result.is_ok());

    // Verify it's gone
    assert!(!cache_dir.exists());
}

#[test]
fn test_download_to_creates_parent_dirs() {
    // This test verifies the parent dir creation logic in download_to
    // We can't test full download without network, but we can test
    // that cache_path_for produces correct paths
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let dataset = HfDataset::builder("download-to-test")
        .cache_dir(temp_dir.path())
        .subset("custom")
        .split("validation")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Verify parquet path building
    assert_eq!(dataset.build_parquet_path(), "custom/validation.parquet");

    // Verify cache path building
    let cache_path = dataset.cache_path_for("custom/validation.parquet");
    assert!(cache_path.to_string_lossy().contains("download-to-test"));
    assert!(cache_path.to_string_lossy().contains("custom"));
}

#[test]
fn test_dataset_info_with_all_fields() {
    let info = DatasetInfo {
        repo_id: "full-test".to_string(),
        splits: vec![
            "train".to_string(),
            "validation".to_string(),
            "test".to_string(),
        ],
        subsets: vec!["default".to_string(), "extra".to_string()],
        download_size: Some(1_000_000),
        description: Some("A comprehensive test dataset for validation".to_string()),
    };

    assert_eq!(info.repo_id, "full-test");
    assert_eq!(info.splits.len(), 3);
    assert_eq!(info.subsets.len(), 2);
    assert_eq!(info.download_size, Some(1_000_000));
    assert!(info.description.is_some());
}

#[test]
fn test_builder_chain_all_methods() {
    let dataset = HfDataset::builder("chain-test")
        .revision("v2.0.0")
        .subset("subset-a")
        .split("test")
        .cache_dir("/custom/cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "chain-test");
    assert_eq!(dataset.revision(), "v2.0.0");
    assert_eq!(dataset.subset(), Some("subset-a"));
    assert_eq!(dataset.split(), Some("test"));
    assert_eq!(dataset.cache_dir(), Path::new("/custom/cache"));
}

#[test]
fn test_deeply_nested_cache_path() {
    let dataset = HfDataset::builder("org/deep/nested/repo")
        .cache_dir("/root")
        .subset("config-name")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("config-name/train.parquet");
    assert!(cache_path
        .to_string_lossy()
        .contains("org/deep/nested/repo"));
}

// ========================================================================
// DatasetCardValidator tests
// ========================================================================

#[test]
fn test_validate_valid_readme() {
    let readme = r"---
license: mit
task_categories:
  - translation
language:
  - en
---
# My Dataset
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_invalid_task_category() {
    let readme = r"---
license: mit
task_categories:
  - code-generation
---
# My Dataset
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].field, "task_categories");
    assert_eq!(errors[0].value, "code-generation");
    // Suggestions are provided for invalid categories
    // (may include similar categories like text-generation)
}

#[test]
fn test_validate_multiple_invalid_categories() {
    let readme = r"---
task_categories:
  - code-generation
  - image-generation
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 2);
}

#[test]
fn test_validate_valid_size_category() {
    let readme = r"---
size_categories:
  - n<1K
  - 1K<n<10K
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_invalid_size_category() {
    let readme = r"---
size_categories:
  - small
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].field, "size_categories");
}

#[test]
fn test_validate_no_frontmatter() {
    let readme = "# Just a title\n\nNo YAML frontmatter here.";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_empty_frontmatter() {
    let readme = "---\n---\n# Empty frontmatter";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_strict_returns_error() {
    let readme = r"---
task_categories:
  - invalid-category
---
";
    let result = DatasetCardValidator::validate_readme_strict(readme);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("invalid-category"));
}

#[test]
fn test_validate_strict_returns_ok() {
    let readme = r"---
task_categories:
  - translation
  - text-classification
---
";
    let result = DatasetCardValidator::validate_readme_strict(readme);
    assert!(result.is_ok());
}

#[test]
fn test_validation_error_display() {
    let err = ValidationError {
        field: "task_categories".to_string(),
        value: "text2text".to_string(),
        suggestions: vec!["text-generation".to_string()],
    };
    let display = err.to_string();
    assert!(display.contains("task_categories"));
    assert!(display.contains("text2text"));
    assert!(display.contains("Did you mean"));
    assert!(display.contains("text-generation"));
}

#[test]
fn test_validation_error_display_no_suggestions() {
    let err = ValidationError {
        field: "size_categories".to_string(),
        value: "huge".to_string(),
        suggestions: vec![],
    };
    let display = err.to_string();
    assert!(display.contains("size_categories"));
    assert!(!display.contains("Did you mean"));
}

#[test]
fn test_levenshtein_distance() {
    assert_eq!(DatasetCardValidator::levenshtein("", ""), 0);
    assert_eq!(DatasetCardValidator::levenshtein("abc", ""), 3);
    assert_eq!(DatasetCardValidator::levenshtein("", "xyz"), 3);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abc"), 0);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abd"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("text", "test"), 1);
}

#[test]
fn test_suggest_similar_finds_matches() {
    let suggestions = DatasetCardValidator::suggest_similar("text-gen", VALID_TASK_CATEGORIES);
    assert!(!suggestions.is_empty());
    // Should find text-generation
    assert!(suggestions.iter().any(|s| s.contains("text")));
}

#[test]
fn test_all_valid_categories_pass() {
    for cat in VALID_TASK_CATEGORIES {
        let readme = format!("---\ntask_categories:\n  - {}\n---\n", cat);
        let errors = DatasetCardValidator::validate_readme(&readme);
        assert!(errors.is_empty(), "Category '{}' should be valid", cat);
    }
}

#[test]
fn test_all_valid_size_categories_pass() {
    for size in VALID_SIZE_CATEGORIES {
        let readme = format!("---\nsize_categories:\n  - {}\n---\n", size);
        let errors = DatasetCardValidator::validate_readme(&readme);
        assert!(errors.is_empty(), "Size '{}' should be valid", size);
    }
}

// ========================================================================
// HfPublisher Tests - EXTREME TDD
// ========================================================================

#[test]
fn test_hf_publisher_new() {
    let publisher = HfPublisher::new("paiml/test-dataset");
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_with_private() {
    let publisher = HfPublisher::new("paiml/test-dataset").with_private(true);
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
    // Private flag is stored internally
}

#[test]
fn test_hf_publisher_with_commit_message() {
    let publisher = HfPublisher::new("paiml/test-dataset").with_commit_message("Test commit");
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_builder_basic() {
    let publisher = HfPublisherBuilder::new("paiml/test-dataset").build();
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_builder_with_all_options() {
    let publisher = HfPublisherBuilder::new("paiml/test-dataset")
        .token("test-token")
        .private(true)
        .commit_message("Custom message")
        .build();
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_parse_org_name_with_slash() {
    // Test that org/name parsing works correctly
    let repo_id = "paiml/python-doctest-corpus";
    let slash_pos = repo_id.find('/');
    assert!(slash_pos.is_some());
    let (org, name) = if let Some(pos) = slash_pos {
        (&repo_id[..pos], &repo_id[pos + 1..])
    } else {
        ("", repo_id)
    };
    assert_eq!(org, "paiml");
    assert_eq!(name, "python-doctest-corpus");
}

#[test]
fn test_hf_publisher_parse_name_without_slash() {
    // Test that single name (no org) works correctly
    let repo_id = "my-dataset";
    let slash_pos = repo_id.find('/');
    assert!(slash_pos.is_none());
}

#[test]
fn test_hf_publisher_commit_url_format() {
    // Verify the commit API URL format
    let repo_id = "paiml/test-dataset";
    let expected_url = format!("{}/datasets/{}/commit/main", HF_API_URL, repo_id);
    assert_eq!(
        expected_url,
        "https://huggingface.co/api/datasets/paiml/test-dataset/commit/main"
    );
}

#[test]
fn test_hf_publisher_create_repo_url_format() {
    // Verify the create repo API URL format
    let expected_url = format!("{}/repos/create", HF_API_URL);
    assert_eq!(expected_url, "https://huggingface.co/api/repos/create");
}

// ========================================================================
// NDJSON Upload API Tests - EXTREME TDD
// These tests define the expected API format BEFORE implementation
// ========================================================================

#[test]
fn test_ndjson_header_format() {
    // HuggingFace commit API expects header as first NDJSON line
    let commit_message = "Upload test file";
    let header = serde_json::json!({
        "key": "header",
        "value": {
            "summary": commit_message,
            "description": ""
        }
    });

    let header_str = header.to_string();
    assert!(header_str.contains("\"key\":\"header\""));
    assert!(header_str.contains("\"summary\":\"Upload test file\""));
}

#[test]
fn test_ndjson_file_operation_format() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // HuggingFace commit API expects file operations with base64 content
    let test_data = b"test file content";
    let content_base64 = STANDARD.encode(test_data);
    let path_in_repo = "data/test.parquet";

    let file_op = serde_json::json!({
        "key": "file",
        "value": {
            "content": content_base64,
            "path": path_in_repo,
            "encoding": "base64"
        }
    });

    let file_str = file_op.to_string();
    assert!(file_str.contains("\"key\":\"file\""));
    assert!(file_str.contains("\"encoding\":\"base64\""));
    assert!(file_str.contains("\"path\":\"data/test.parquet\""));
}

#[test]
fn test_ndjson_payload_is_newline_delimited() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // NDJSON = one JSON object per line, separated by newlines
    let header = serde_json::json!({
        "key": "header",
        "value": {"summary": "Test", "description": ""}
    });

    let file_op = serde_json::json!({
        "key": "file",
        "value": {
            "content": STANDARD.encode(b"data"),
            "path": "test.txt",
            "encoding": "base64"
        }
    });

    let ndjson = format!("{}\n{}", header, file_op);

    // Should have exactly 2 lines
    let lines: Vec<&str> = ndjson.lines().collect();
    assert_eq!(lines.len(), 2);

    // Each line should be valid JSON
    assert!(serde_json::from_str::<serde_json::Value>(lines[0]).is_ok());
    assert!(serde_json::from_str::<serde_json::Value>(lines[1]).is_ok());
}

#[test]
fn test_build_ndjson_payload() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // Test the helper function that builds the complete NDJSON payload
    let commit_message = "Upload via alimentar";
    let path_in_repo = "train.parquet";
    let data = b"parquet binary data here";

    let payload = build_ndjson_upload_payload(commit_message, path_in_repo, data);

    // Verify structure
    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2, "NDJSON should have exactly 2 lines");

    // Parse and verify header
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["key"], "header");
    assert_eq!(header["value"]["summary"], commit_message);

    // Parse and verify file operation
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["key"], "file");
    assert_eq!(file_op["value"]["path"], path_in_repo);
    assert_eq!(file_op["value"]["encoding"], "base64");

    // Verify content round-trips correctly
    let encoded_content = file_op["value"]["content"].as_str().unwrap();
    let decoded = STANDARD.decode(encoded_content).unwrap();
    assert_eq!(decoded, data);
}

// ========================================================================
// LFS Preupload API Tests - EXTREME TDD for binary file support
// These tests define the expected API format BEFORE implementation
// ========================================================================

#[test]
fn test_is_binary_file_detection() {
    // Binary file extensions that require LFS
    assert!(is_binary_file("train.parquet"));
    assert!(is_binary_file("data.arrow"));
    assert!(is_binary_file("image.png"));
    assert!(is_binary_file("model.bin"));
    assert!(is_binary_file("weights.safetensors"));

    // Text files that can use direct commit
    assert!(!is_binary_file("README.md"));
    assert!(!is_binary_file("config.json"));
    assert!(!is_binary_file("data.csv"));
    assert!(!is_binary_file(".gitattributes"));
}

#[test]
fn test_compute_sha256_for_lfs() {
    // LFS requires SHA256 hash of file content
    let data = b"test content for hashing";
    let hash = compute_sha256(data);

    // SHA256 should be 64 hex characters
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

    // Same input should produce same hash
    assert_eq!(hash, compute_sha256(data));

    // Different input should produce different hash
    assert_ne!(hash, compute_sha256(b"different content"));
}

#[test]
fn test_build_lfs_preupload_request() {
    // HuggingFace preupload API request format
    let path = "data/train.parquet";
    let data = b"parquet binary content";

    let request = build_lfs_preupload_request(path, data);

    // Parse and verify structure
    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert!(json.get("files").is_some());

    let files = json["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);

    let file = &files[0];
    assert_eq!(file["path"], path);
    assert_eq!(file["size"], data.len());
    // Sample is first 512 bytes base64 encoded
    assert!(file["sample"].is_string());
}

#[test]
fn test_build_ndjson_lfs_commit() {
    // LFS commit uses lfsFile key with OID instead of base64 content
    let commit_message = "Upload parquet via LFS";
    let path_in_repo = "data/train.parquet";
    let oid = "abc123def456"; // SHA256 hash
    let size = 1024usize;

    let payload = build_ndjson_lfs_commit(commit_message, path_in_repo, oid, size);

    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2, "NDJSON should have exactly 2 lines");

    // Header should be same format
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["key"], "header");
    assert_eq!(header["value"]["summary"], commit_message);

    // File should use lfsFile key with OID
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["key"], "lfsFile");
    assert_eq!(file_op["value"]["path"], path_in_repo);
    assert_eq!(file_op["value"]["algo"], "sha256");
    assert_eq!(file_op["value"]["oid"], oid);
    assert_eq!(file_op["value"]["size"], size);
}

#[test]
fn test_lfs_preupload_url_format() {
    // Verify preupload API URL format
    let repo_id = "paiml/test-dataset";
    let expected_url = format!("{}/datasets/{}/preupload/main", HF_API_URL, repo_id);
    assert_eq!(
        expected_url,
        "https://huggingface.co/api/datasets/paiml/test-dataset/preupload/main"
    );
}

#[test]
fn test_lfs_batch_api_url_format() {
    // LFS batch API uses different URL format (Git LFS standard)
    let repo_id = "paiml/test-dataset";
    let expected_url = format!(
        "https://huggingface.co/datasets/{}.git/info/lfs/objects/batch",
        repo_id
    );
    assert_eq!(
        expected_url,
        "https://huggingface.co/datasets/paiml/test-dataset.git/info/lfs/objects/batch"
    );
}

#[test]
fn test_build_lfs_batch_request() {
    // LFS batch API request format (Git LFS standard)
    let oid = "abc123def456";
    let size = 1024usize;

    let request = build_lfs_batch_request(oid, size);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["operation"], "upload");
    assert!(json["transfers"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("basic")));

    let objects = json["objects"].as_array().unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0]["oid"], oid);
    assert_eq!(objects[0]["size"], size);
}

// ========== DatasetCardValidator Tests (GH-6) ==========

#[test]
fn test_valid_task_categories() {
    assert!(DatasetCardValidator::is_valid_task_category(
        "text-generation"
    ));
    assert!(DatasetCardValidator::is_valid_task_category("translation"));
    assert!(DatasetCardValidator::is_valid_task_category(
        "text-classification"
    ));
    assert!(DatasetCardValidator::is_valid_task_category(
        "question-answering"
    ));
    assert!(DatasetCardValidator::is_valid_task_category(
        "text2text-generation"
    ));
}

#[test]
fn test_invalid_task_categories() {
    assert!(!DatasetCardValidator::is_valid_task_category(
        "code-generation"
    ));
    assert!(!DatasetCardValidator::is_valid_task_category(
        "invalid-task"
    ));
    assert!(!DatasetCardValidator::is_valid_task_category(""));
}

#[test]
fn test_valid_licenses() {
    assert!(DatasetCardValidator::is_valid_license("apache-2.0"));
    assert!(DatasetCardValidator::is_valid_license("mit"));
    assert!(DatasetCardValidator::is_valid_license("Apache-2.0")); // Case insensitive
    assert!(DatasetCardValidator::is_valid_license("MIT"));
}

#[test]
fn test_invalid_licenses() {
    assert!(!DatasetCardValidator::is_valid_license("invalid-license"));
    assert!(!DatasetCardValidator::is_valid_license(""));
}

#[test]
fn test_valid_size_categories() {
    assert!(DatasetCardValidator::is_valid_size_category("n<1K"));
    assert!(DatasetCardValidator::is_valid_size_category("1K<n<10K"));
    assert!(DatasetCardValidator::is_valid_size_category("n>1T"));
}

#[test]
fn test_invalid_size_categories() {
    assert!(!DatasetCardValidator::is_valid_size_category("small"));
    assert!(!DatasetCardValidator::is_valid_size_category("1000"));
}

#[test]
fn test_suggest_task_category() {
    assert_eq!(
        DatasetCardValidator::suggest_task_category("code-generation"),
        Some("text-generation")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("qa"),
        Some("question-answering")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("ner"),
        Some("token-classification")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("sentiment"),
        Some("text-classification")
    );
}
