//! HTTP/HTTPS storage backend (read-only).
//!
//! Provides read-only access to datasets hosted on HTTP/HTTPS servers.
//! Useful for accessing public datasets without requiring cloud credentials.

use bytes::Bytes;
use reqwest::{
    blocking::Client,
    header::{CONTENT_LENGTH, RANGE},
};

use super::StorageBackend;
use crate::error::{Error, Result};

/// A read-only storage backend using HTTP/HTTPS.
///
/// This backend is designed for accessing publicly hosted datasets
/// over HTTP/HTTPS. It supports range requests for efficient partial
/// reads when the server supports them.
///
/// # Limitations
///
/// - Read-only: `put` and `delete` operations will return errors
/// - `list` is not supported (HTTP doesn't have directory listings)
///
/// # Example
///
/// ```no_run
/// use alimentar::backend::{HttpBackend, StorageBackend};
///
/// let backend = HttpBackend::new("https://huggingface.co/datasets").unwrap();
/// let data = backend.get("squad/train.parquet").unwrap();
/// ```
#[derive(Debug)]
pub struct HttpBackend {
    client: Client,
    base_url: String,
}

impl HttpBackend {
    /// Creates a new HTTP backend with the given base URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for all requests. Keys will be appended to this.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    /// Creates a new HTTP backend with custom client configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for all requests
    /// * `timeout_secs` - Request timeout in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn with_timeout(base_url: impl Into<String>, timeout_secs: u64) -> Result<Self> {
        let base_url = base_url.into();
        let client = Client::builder()
            .user_agent("alimentar/0.1.0")
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| Error::storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self { client, base_url })
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Constructs the full URL for a key.
    fn url_for(&self, key: &str) -> String {
        if self.base_url.ends_with('/') {
            format!("{}{}", self.base_url, key)
        } else {
            format!("{}/{}", self.base_url, key)
        }
    }
}

impl StorageBackend for HttpBackend {
    fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        // HTTP doesn't support directory listings
        Err(Error::storage(
            "HTTP backend does not support listing (use a specific key instead)",
        ))
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        let url = self.url_for(key);

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP GET failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(bytes)
    }

    fn put(&self, key: &str, _data: Bytes) -> Result<()> {
        Err(Error::storage(format!(
            "HTTP backend is read-only, cannot write to '{}'",
            key
        )))
    }

    fn delete(&self, key: &str) -> Result<()> {
        Err(Error::storage(format!(
            "HTTP backend is read-only, cannot delete '{}'",
            key
        )))
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let url = self.url_for(key);

        let response = self
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        Ok(response.status().is_success())
    }

    fn size(&self, key: &str) -> Result<u64> {
        let url = self.url_for(key);

        let response = self
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP HEAD failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        // Try to get Content-Length header
        if let Some(content_length) = response.headers().get(CONTENT_LENGTH) {
            if let Ok(len_str) = content_length.to_str() {
                if let Ok(len) = len_str.parse::<u64>() {
                    return Ok(len);
                }
            }
        }

        Err(Error::storage(format!(
            "Server did not provide Content-Length for '{}'",
            url
        )))
    }
}

/// HTTP backend with support for partial/range requests.
///
/// This variant supports reading specific byte ranges, which is useful
/// for large files when only a portion is needed.
#[derive(Debug)]
pub struct RangeHttpBackend {
    inner: HttpBackend,
}

impl RangeHttpBackend {
    /// Creates a new range-capable HTTP backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the base URL is invalid.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            inner: HttpBackend::new(base_url)?,
        })
    }

    /// Reads a specific byte range from a key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to read from
    /// * `start` - Starting byte offset (inclusive)
    /// * `end` - Ending byte offset (inclusive)
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the server doesn't support
    /// ranges.
    pub fn get_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = self.inner.url_for(key);
        let range_header = format!("bytes={}-{}", start, end);

        let response = self
            .inner
            .client
            .get(&url)
            .header(RANGE, range_header)
            .send()
            .map_err(|e| Error::storage(format!("HTTP GET range error for '{}': {}", url, e)))?;

        // 206 Partial Content is expected for range requests
        if response.status().as_u16() != 206 && !response.status().is_success() {
            return Err(Error::storage(format!(
                "HTTP GET range failed for '{}': status {}",
                url,
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| Error::storage(format!("Failed to read HTTP response body: {e}")))?;

        Ok(bytes)
    }

    /// Checks if the server supports range requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP HEAD request fails.
    pub fn supports_range(&self, key: &str) -> Result<bool> {
        let url = self.inner.url_for(key);

        let response = self
            .inner
            .client
            .head(&url)
            .send()
            .map_err(|e| Error::storage(format!("HTTP HEAD error for '{}': {}", url, e)))?;

        if !response.status().is_success() {
            return Ok(false);
        }

        // Check for Accept-Ranges header
        if let Some(accept_ranges) = response.headers().get("accept-ranges") {
            if let Ok(value) = accept_ranges.to_str() {
                return Ok(value != "none");
            }
        }

        Ok(false)
    }
}

impl StorageBackend for RangeHttpBackend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        self.inner.get(key)
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        self.inner.put(key, data)
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key)
    }

    fn size(&self, key: &str) -> Result<u64> {
        self.inner.size(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_construction() {
        let backend = HttpBackend::new("https://example.com/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("file.txt"),
            "https://example.com/data/file.txt"
        );

        let backend_slash = HttpBackend::new("https://example.com/data/")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend_slash.url_for("file.txt"),
            "https://example.com/data/file.txt"
        );
    }

    #[test]
    fn test_base_url() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(backend.base_url(), "https://example.com");
    }

    #[test]
    fn test_put_is_read_only() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("test.txt", Bytes::from("data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_is_read_only() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_not_supported() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("");
        assert!(result.is_err());
    }

    #[test]
    fn test_with_timeout() {
        let backend = HttpBackend::with_timeout("https://example.com", 30);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_range_http_backend_new() {
        let backend = RangeHttpBackend::new("https://example.com");
        assert!(backend.is_ok());
    }

    #[test]
    fn test_range_http_backend_list_not_supported() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.list("");
        assert!(result.is_err());
    }

    #[test]
    fn test_range_http_backend_put_is_read_only() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.put("test.txt", Bytes::from("data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_range_http_backend_delete_is_read_only() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let result = backend.delete("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_url_construction_nested_path() {
        let backend = HttpBackend::new("https://example.com/api/v1/data")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        assert_eq!(
            backend.url_for("datasets/train.parquet"),
            "https://example.com/api/v1/data/datasets/train.parquet"
        );
    }

    #[test]
    fn test_http_backend_debug() {
        let backend = HttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("HttpBackend"));
        assert!(debug_str.contains("example.com"));
    }

    #[test]
    fn test_range_http_backend_debug() {
        let backend = RangeHttpBackend::new("https://example.com")
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("RangeHttpBackend"));
    }
}
