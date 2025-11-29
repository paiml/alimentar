//! HuggingFace Hub dataset importer.
//!
//! Provides functionality to import datasets from the HuggingFace Hub.
//! Supports downloading parquet files directly from HF datasets.
//!
//! # Example
//!
//! ```no_run
//! use alimentar::{hf_hub::HfDataset, Dataset};
//!
//! // Import a dataset from HuggingFace Hub
//! let hf = HfDataset::builder("squad")
//!     .revision("main")
//!     .split("train")
//!     .build()
//!     .unwrap();
//!
//! let dataset = hf.download().unwrap();
//! println!("Loaded {} rows", dataset.len());
//! ```

use std::path::{Path, PathBuf};

use crate::{
    backend::{HttpBackend, StorageBackend},
    dataset::ArrowDataset,
    error::{Error, Result},
};

/// Base URL for HuggingFace Hub datasets API.
const HF_HUB_URL: &str = "https://huggingface.co";

/// HuggingFace Hub dataset configuration and loader.
///
/// This struct provides a builder pattern for configuring and downloading
/// datasets from the HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct HfDataset {
    /// Dataset repository name (e.g., "squad", "glue", "openai/gsm8k")
    repo_id: String,
    /// Git revision (branch, tag, or commit hash)
    revision: String,
    /// Dataset subset/config (optional)
    subset: Option<String>,
    /// Data split (train, validation, test)
    split: Option<String>,
    /// Local cache directory
    cache_dir: PathBuf,
}

impl HfDataset {
    /// Creates a new builder for a HuggingFace dataset.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - The dataset repository ID (e.g., "squad", "openai/gsm8k")
    pub fn builder(repo_id: impl Into<String>) -> HfDatasetBuilder {
        HfDatasetBuilder::new(repo_id)
    }

    /// Returns the repository ID.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Returns the revision being used.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the subset/config if set.
    pub fn subset(&self) -> Option<&str> {
        self.subset.as_deref()
    }

    /// Returns the split if set.
    pub fn split(&self) -> Option<&str> {
        self.split.as_deref()
    }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Downloads the dataset and returns an ArrowDataset.
    ///
    /// This method:
    /// 1. Checks the local cache for existing data
    /// 2. Downloads parquet files from HuggingFace Hub if not cached
    /// 3. Loads the parquet files into an ArrowDataset
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dataset cannot be found on HuggingFace Hub
    /// - The download fails
    /// - The parquet files cannot be parsed
    pub fn download(&self) -> Result<ArrowDataset> {
        // Build the URL path for the parquet file
        let parquet_path = self.build_parquet_path();
        let cache_file = self.cache_path_for(&parquet_path);

        // Check cache first
        if cache_file.exists() {
            return ArrowDataset::from_parquet(&cache_file);
        }

        // Download from HF Hub
        let url = self.build_download_url(&parquet_path);
        let http = HttpBackend::with_timeout(&url, 300)?;

        // The key is empty since we've already built the full URL
        let data = http.get("")?;

        // Ensure cache directory exists
        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(e, parent))?;
        }

        // Write to cache
        std::fs::write(&cache_file, &data).map_err(|e| Error::io(e, &cache_file))?;

        // Load from cache
        ArrowDataset::from_parquet(&cache_file)
    }

    /// Downloads the dataset to a specific output path.
    ///
    /// # Arguments
    ///
    /// * `output` - Path where the dataset should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the download or save fails.
    pub fn download_to(&self, output: impl AsRef<Path>) -> Result<ArrowDataset> {
        let output = output.as_ref();
        let parquet_path = self.build_parquet_path();
        let url = self.build_download_url(&parquet_path);

        let http = HttpBackend::with_timeout(&url, 300)?;
        let data = http.get("")?;

        // Ensure parent directory exists
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(e, parent))?;
        }

        // Write to output
        std::fs::write(output, &data).map_err(|e| Error::io(e, output))?;

        // Load and return
        ArrowDataset::from_parquet(output)
    }

    /// Builds the parquet file path within the repository.
    fn build_parquet_path(&self) -> String {
        let mut path_parts = Vec::new();

        // Add subset/config if present
        if let Some(ref subset) = self.subset {
            path_parts.push(subset.clone());
        } else {
            path_parts.push("default".to_string());
        }

        // Add split
        let split = self.split.as_deref().unwrap_or("train");
        path_parts.push(format!("{split}.parquet"));

        path_parts.join("/")
    }

    /// Builds the download URL for a parquet file.
    fn build_download_url(&self, parquet_path: &str) -> String {
        format!(
            "{}/datasets/{}/resolve/{}/data/{}",
            HF_HUB_URL, self.repo_id, self.revision, parquet_path
        )
    }

    /// Returns the cache path for a given parquet path.
    fn cache_path_for(&self, parquet_path: &str) -> PathBuf {
        self.cache_dir
            .join("huggingface")
            .join("datasets")
            .join(&self.repo_id)
            .join(&self.revision)
            .join(parquet_path)
    }

    /// Clears the local cache for this dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache cannot be deleted.
    pub fn clear_cache(&self) -> Result<()> {
        let cache_path = self
            .cache_dir
            .join("huggingface")
            .join("datasets")
            .join(&self.repo_id);

        if cache_path.exists() {
            std::fs::remove_dir_all(&cache_path).map_err(|e| Error::io(e, &cache_path))?;
        }

        Ok(())
    }
}

/// Builder for configuring HuggingFace dataset downloads.
#[derive(Debug, Clone)]
pub struct HfDatasetBuilder {
    repo_id: String,
    revision: String,
    subset: Option<String>,
    split: Option<String>,
    cache_dir: Option<PathBuf>,
}

impl HfDatasetBuilder {
    /// Creates a new builder with the given repository ID.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: "main".to_string(),
            subset: None,
            split: None,
            cache_dir: None,
        }
    }

    /// Sets the Git revision (branch, tag, or commit hash).
    ///
    /// Default is "main".
    #[must_use]
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Sets the dataset subset/configuration.
    ///
    /// Some datasets have multiple configurations (e.g., "glue" has "cola",
    /// "sst2", etc.)
    #[must_use]
    pub fn subset(mut self, subset: impl Into<String>) -> Self {
        self.subset = Some(subset.into());
        self
    }

    /// Sets the data split to download.
    ///
    /// Common values: "train", "validation", "test"
    #[must_use]
    pub fn split(mut self, split: impl Into<String>) -> Self {
        self.split = Some(split.into());
        self
    }

    /// Sets the local cache directory.
    ///
    /// Default is `~/.cache/alimentar` on Unix or `%LOCALAPPDATA%/alimentar` on
    /// Windows.
    #[must_use]
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Builds the HfDataset configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or invalid.
    pub fn build(self) -> Result<HfDataset> {
        if self.repo_id.is_empty() {
            return Err(Error::invalid_config("Repository ID cannot be empty"));
        }

        let cache_dir = self.cache_dir.unwrap_or_else(default_cache_dir);

        Ok(HfDataset {
            repo_id: self.repo_id,
            revision: self.revision,
            subset: self.subset,
            split: self.split,
            cache_dir,
        })
    }
}

/// Returns the default cache directory for the current platform.
fn default_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local_app_data)
                .join("alimentar")
                .join("cache");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg_cache).join("alimentar");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("alimentar");
        }
    }

    // Fallback to temp directory
    std::env::temp_dir().join("alimentar").join("cache")
}

/// Lists available parquet files for a dataset on HuggingFace Hub.
///
/// This function queries the HuggingFace Hub API to list available
/// parquet files for a given dataset.
///
/// # Arguments
///
/// * `repo_id` - The dataset repository ID
/// * `revision` - Git revision (default "main")
///
/// # Errors
///
/// Returns an error if the HTTP request fails or the response cannot be parsed.
///
/// # Note
///
/// This function requires the HF Hub API and may be rate-limited.
pub fn list_dataset_files(repo_id: &str, revision: Option<&str>) -> Result<Vec<String>> {
    let revision = revision.unwrap_or("main");
    let url = format!("{}/api/datasets/{}/tree/{}", HF_HUB_URL, repo_id, revision);

    let http = HttpBackend::with_timeout(&url, 30)?;
    let data = http.get("")?;

    // Parse JSON response
    let json: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| Error::storage(format!("Failed to parse HF Hub response: {e}")))?;

    let mut parquet_files = Vec::new();

    if let Some(items) = json.as_array() {
        for item in items {
            if let Some(path) = item.get("path").and_then(|p| p.as_str()) {
                if path.ends_with(".parquet") {
                    parquet_files.push(path.to_string());
                }
            }
        }
    }

    Ok(parquet_files)
}

/// Information about a HuggingFace dataset.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Dataset repository ID
    pub repo_id: String,
    /// Available splits
    pub splits: Vec<String>,
    /// Available subsets/configs
    pub subsets: Vec<String>,
    /// Total download size in bytes (if known)
    pub download_size: Option<u64>,
    /// Description
    pub description: Option<String>,
}

// ============================================================================
// HuggingFace Hub Publisher (Upload Support)
// ============================================================================

/// HuggingFace Hub API URL for uploads
const HF_API_URL: &str = "https://huggingface.co/api";

/// Publisher for uploading datasets to HuggingFace Hub.
///
/// # Example
///
/// ```no_run
/// use alimentar::hf_hub::HfPublisher;
/// use arrow::record_batch::RecordBatch;
///
/// let publisher = HfPublisher::new("paiml/my-dataset")
///     .with_token(std::env::var("HF_TOKEN").unwrap())
///     .with_private(false);
///
/// // publisher.upload_parquet("train.parquet", &batch).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HfPublisher {
    /// Repository ID (e.g., "paiml/depyler-citl")
    repo_id: String,
    /// HuggingFace API token
    token: Option<String>,
    /// Whether the dataset should be private
    private: bool,
    /// Commit message for uploads
    commit_message: String,
}

impl HfPublisher {
    /// Creates a new publisher for a HuggingFace dataset repository.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: std::env::var("HF_TOKEN").ok(),
            private: false,
            commit_message: "Upload via alimentar".to_string(),
        }
    }

    /// Sets the HuggingFace API token.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets whether the dataset should be private.
    #[must_use]
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Sets the commit message for uploads.
    #[must_use]
    pub fn with_commit_message(mut self, message: impl Into<String>) -> Self {
        self.commit_message = message.into();
        self
    }

    /// Returns the repository ID.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Creates the dataset repository on HuggingFace Hub if it doesn't exist.
    #[cfg(feature = "http")]
    pub async fn create_repo(&self) -> Result<()> {
        let token = self.token.as_ref().ok_or_else(|| {
            Error::io_no_path(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "HF_TOKEN required for upload",
            ))
        })?;

        let client = reqwest::Client::new();
        let url = format!("{}/repos/create", HF_API_URL);

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&serde_json::json!({
                "type": "dataset",
                "name": self.repo_id,
                "private": self.private
            }))
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        // 409 Conflict means repo already exists, which is fine
        if response.status().is_success() || response.status().as_u16() == 409 {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(Error::io_no_path(std::io::Error::other(format!(
                "Failed to create repo: {} - {}",
                status, body
            ))))
        }
    }

    /// Uploads a parquet file to the repository.
    #[cfg(feature = "http")]
    pub async fn upload_file(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        let token = self.token.as_ref().ok_or_else(|| {
            Error::io_no_path(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "HF_TOKEN required for upload",
            ))
        })?;

        let client = reqwest::Client::new();
        let url = format!(
            "{}/datasets/{}/upload/main/{}",
            HF_HUB_URL, self.repo_id, path_in_repo
        );

        let response = client
            .put(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/octet-stream")
            .body(data.to_vec())
            .send()
            .await
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(Error::io_no_path(std::io::Error::other(format!(
                "Failed to upload: {} - {}",
                status, body
            ))))
        }
    }

    /// Uploads a RecordBatch as a parquet file.
    #[cfg(feature = "http")]
    pub async fn upload_batch(
        &self,
        path_in_repo: &str,
        batch: &arrow::record_batch::RecordBatch,
    ) -> Result<()> {
        use parquet::arrow::ArrowWriter;

        // Write batch to parquet in memory
        let mut buffer = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buffer, batch.schema(), None).map_err(Error::Parquet)?;
            writer.write(batch).map_err(Error::Parquet)?;
            writer.close().map_err(Error::Parquet)?;
        }

        self.upload_file(path_in_repo, &buffer).await
    }

    /// Uploads a local parquet file to the repository.
    #[cfg(feature = "http")]
    pub async fn upload_parquet_file(&self, local_path: &Path, path_in_repo: &str) -> Result<()> {
        let data = std::fs::read(local_path).map_err(|e| Error::io(e, local_path))?;
        self.upload_file(path_in_repo, &data).await
    }

    /// Synchronous wrapper for creating repo (for CLI use).
    #[cfg(all(feature = "http", feature = "tokio-runtime"))]
    pub fn create_repo_sync(&self) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.create_repo())
    }

    /// Synchronous wrapper for uploading file (for CLI use).
    #[cfg(all(feature = "http", feature = "tokio-runtime"))]
    pub fn upload_file_sync(&self, path_in_repo: &str, data: &[u8]) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_file(path_in_repo, data))
    }

    /// Synchronous wrapper for uploading parquet file (for CLI use).
    #[cfg(all(feature = "http", feature = "tokio-runtime"))]
    pub fn upload_parquet_file_sync(&self, local_path: &Path, path_in_repo: &str) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_parquet_file(local_path, path_in_repo))
    }

    /// Uploads a README.md with validation.
    ///
    /// Validates the dataset card metadata before upload to catch issues like
    /// invalid `task_categories` before they cause HuggingFace warnings.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails or upload fails.
    #[cfg(feature = "http")]
    pub async fn upload_readme_validated(&self, content: &str) -> Result<()> {
        DatasetCardValidator::validate_readme_strict(content)?;
        self.upload_file("README.md", content.as_bytes()).await
    }

    /// Synchronous wrapper for validated README upload.
    #[cfg(all(feature = "http", feature = "tokio-runtime"))]
    pub fn upload_readme_validated_sync(&self, content: &str) -> Result<()> {
        tokio::runtime::Runtime::new()
            .map_err(|e| Error::io_no_path(std::io::Error::other(e)))?
            .block_on(self.upload_readme_validated(content))
    }
}

/// Builder for HfPublisher with fluent interface.
#[derive(Debug, Clone)]
pub struct HfPublisherBuilder {
    repo_id: String,
    token: Option<String>,
    private: bool,
    commit_message: String,
}

impl HfPublisherBuilder {
    /// Creates a new builder.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            token: None,
            private: false,
            commit_message: "Upload via alimentar".to_string(),
        }
    }

    /// Sets the token.
    #[must_use]
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets private flag.
    #[must_use]
    pub fn private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Sets commit message.
    #[must_use]
    pub fn commit_message(mut self, message: impl Into<String>) -> Self {
        self.commit_message = message.into();
        self
    }

    /// Builds the publisher.
    pub fn build(self) -> HfPublisher {
        HfPublisher {
            repo_id: self.repo_id,
            token: self.token.or_else(|| std::env::var("HF_TOKEN").ok()),
            private: self.private,
            commit_message: self.commit_message,
        }
    }
}

// ============================================================================
// Dataset Card Validation
// ============================================================================

/// Valid HuggingFace task categories as of 2024.
/// Source: https://huggingface.co/docs/hub/datasets-cards#task-categories
pub const VALID_TASK_CATEGORIES: &[&str] = &[
    // NLP
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "feature-extraction",
    "text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    // Computer Vision
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    // Multimodal
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "graph-ml",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "image-feature-extraction",
    // Other
    "other",
];

/// Valid HuggingFace size categories.
pub const VALID_SIZE_CATEGORIES: &[&str] = &[
    "n<1K",
    "1K<n<10K",
    "10K<n<100K",
    "100K<n<1M",
    "1M<n<10M",
    "10M<n<100M",
    "100M<n<1B",
    "1B<n<10B",
    "10B<n<100B",
    "100B<n<1T",
    "n>1T",
];

/// Validation error for dataset card metadata.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The field that has an invalid value
    pub field: String,
    /// The invalid value
    pub value: String,
    /// Suggested valid values (if applicable)
    pub suggestions: Vec<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid '{}': '{}' is not valid", self.field, self.value)?;
        if !self.suggestions.is_empty() {
            write!(f, ". Did you mean: {}?", self.suggestions.join(", "))?;
        }
        Ok(())
    }
}

/// Validator for HuggingFace dataset card YAML metadata.
///
/// Validates common fields against HuggingFace's official accepted values.
///
/// # Example
///
/// ```
/// use alimentar::hf_hub::DatasetCardValidator;
///
/// let readme = r"---
/// license: mit
/// task_categories:
///   - translation
/// ---
/// # My Dataset
/// ";
///
/// let errors = DatasetCardValidator::validate_readme(readme);
/// assert!(errors.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct DatasetCardValidator;

impl DatasetCardValidator {
    /// Validates a README.md content and returns any validation errors.
    ///
    /// Parses the YAML frontmatter and validates:
    /// - `task_categories`: Must be from the official HuggingFace list
    /// - `size_categories`: Must match the HuggingFace format
    pub fn validate_readme(content: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Extract YAML frontmatter (between --- markers)
        let Some(yaml_content) = Self::extract_frontmatter(content) else {
            return errors;
        };

        // Parse YAML
        let Ok(yaml) = serde_yaml::from_str::<serde_yaml::Value>(&yaml_content) else {
            return errors;
        };

        // Validate task_categories
        if let Some(categories) = yaml.get("task_categories") {
            if let Some(arr) = categories.as_sequence() {
                for cat in arr {
                    if let Some(cat_str) = cat.as_str() {
                        if !VALID_TASK_CATEGORIES.contains(&cat_str) {
                            errors.push(ValidationError {
                                field: "task_categories".to_string(),
                                value: cat_str.to_string(),
                                suggestions: Self::suggest_similar(cat_str, VALID_TASK_CATEGORIES),
                            });
                        }
                    }
                }
            }
        }

        // Validate size_categories
        if let Some(sizes) = yaml.get("size_categories") {
            if let Some(arr) = sizes.as_sequence() {
                for size in arr {
                    if let Some(size_str) = size.as_str() {
                        if !VALID_SIZE_CATEGORIES.contains(&size_str) {
                            errors.push(ValidationError {
                                field: "size_categories".to_string(),
                                value: size_str.to_string(),
                                suggestions: Self::suggest_similar(size_str, VALID_SIZE_CATEGORIES),
                            });
                        }
                    }
                }
            }
        }

        errors
    }

    /// Validates a README file and returns a Result.
    ///
    /// Returns Ok(()) if valid, or Err with combined error messages.
    pub fn validate_readme_strict(content: &str) -> Result<()> {
        let errors = Self::validate_readme(content);
        if errors.is_empty() {
            Ok(())
        } else {
            let msg = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            Err(Error::invalid_config(msg))
        }
    }

    /// Extracts YAML frontmatter from markdown content.
    fn extract_frontmatter(content: &str) -> Option<String> {
        let content = content.trim_start();
        if !content.starts_with("---") {
            return None;
        }

        let rest = &content[3..];
        let end_idx = rest.find("\n---")?;
        Some(rest[..end_idx].to_string())
    }

    /// Suggests similar valid values using simple substring matching.
    fn suggest_similar(value: &str, valid: &[&str]) -> Vec<String> {
        let value_lower = value.to_lowercase();
        valid
            .iter()
            .filter(|v| {
                let v_lower = v.to_lowercase();
                v_lower.contains(&value_lower)
                    || value_lower.contains(&v_lower)
                    || Self::levenshtein(&value_lower, &v_lower) <= 3
            })
            .take(3)
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Simple Levenshtein distance for fuzzy matching.
    fn levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut dp = vec![vec![0; n + 1]; m + 1];

        for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
            row[0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
            *cell = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }
}

#[cfg(test)]
mod tests {
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
            PathBuf::from("/tmp/cache/huggingface/datasets/squad/main/default/train.parquet")
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
  - text2text-generation
---
# My Dataset
";
        let errors = DatasetCardValidator::validate_readme(readme);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "task_categories");
        assert_eq!(errors[0].value, "text2text-generation");
        // Should suggest "text-generation" as similar
        assert!(errors[0]
            .suggestions
            .contains(&"text-generation".to_string()));
    }

    #[test]
    fn test_validate_multiple_invalid_categories() {
        let readme = r"---
task_categories:
  - text2text-generation
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
}
