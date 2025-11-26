//! In-memory storage backend.

use std::{collections::HashMap, sync::RwLock};

use bytes::Bytes;

use super::StorageBackend;
use crate::error::{Error, Result};

/// An in-memory storage backend.
///
/// Useful for testing and WASM environments where filesystem access
/// is not available. All data is stored in memory and lost when the
/// backend is dropped.
///
/// # Thread Safety
///
/// This backend is thread-safe and can be shared across threads.
///
/// # Example
///
/// ```
/// use alimentar::backend::{MemoryBackend, StorageBackend};
/// use bytes::Bytes;
///
/// let backend = MemoryBackend::new();
/// backend.put("key", Bytes::from("value")).unwrap();
/// let data = backend.get("key").unwrap();
/// assert_eq!(data, Bytes::from("value"));
/// ```
#[derive(Debug, Default)]
pub struct MemoryBackend {
    data: RwLock<HashMap<String, Bytes>>,
}

impl MemoryBackend {
    /// Creates a new empty memory backend.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a memory backend with initial data.
    pub fn with_data(data: HashMap<String, Bytes>) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }

    /// Returns the number of keys stored.
    pub fn len(&self) -> usize {
        self.data.read().map(|d| d.len()).unwrap_or(0)
    }

    /// Returns true if no data is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all stored data.
    pub fn clear(&self) {
        if let Ok(mut data) = self.data.write() {
            data.clear();
        }
    }
}

impl StorageBackend for MemoryBackend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        let keys: Vec<String> = data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        Ok(keys)
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        data.get(key)
            .cloned()
            .ok_or_else(|| Error::storage(format!("Key not found: {}", key)))
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let mut store = self
            .data
            .write()
            .map_err(|_| Error::storage("Failed to acquire write lock"))?;

        store.insert(key.to_string(), data);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| Error::storage("Failed to acquire write lock"))?;

        data.remove(key);
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        Ok(data.contains_key(key))
    }

    fn size(&self, key: &str) -> Result<u64> {
        let data = self
            .data
            .read()
            .map_err(|_| Error::storage("Failed to acquire read lock"))?;

        data.get(key)
            .map(|d| d.len() as u64)
            .ok_or_else(|| Error::storage(format!("Key not found: {}", key)))
    }
}

impl Clone for MemoryBackend {
    fn clone(&self) -> Self {
        let data = self.data.read().map(|d| d.clone()).unwrap_or_default();
        Self::with_data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let backend = MemoryBackend::new();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_put_and_get() {
        let backend = MemoryBackend::new();

        let data = Bytes::from("hello world");
        backend
            .put("key", data.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let retrieved = backend
            .get("key")
            .ok()
            .unwrap_or_else(|| panic!("Should get"));
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_exists() {
        let backend = MemoryBackend::new();

        assert!(!backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));

        backend
            .put("key", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));
    }

    #[test]
    fn test_delete() {
        let backend = MemoryBackend::new();

        backend
            .put("key", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should exist")));

        backend
            .delete("key")
            .ok()
            .unwrap_or_else(|| panic!("Should delete"));

        assert!(!backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should not exist")));
    }

    #[test]
    fn test_list() {
        let backend = MemoryBackend::new();

        backend
            .put("foo/bar", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("foo/baz", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("other", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let foo_keys = backend
            .list("foo/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(foo_keys.len(), 2);

        let all_keys = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_size() {
        let backend = MemoryBackend::new();

        let data = Bytes::from("1234567890"); // 10 bytes
        backend
            .put("key", data)
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let size = backend
            .size("key")
            .ok()
            .unwrap_or_else(|| panic!("Should get size"));
        assert_eq!(size, 10);
    }

    #[test]
    fn test_get_nonexistent() {
        let backend = MemoryBackend::new();
        let result = backend.get("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_len() {
        let backend = MemoryBackend::new();
        assert_eq!(backend.len(), 0);

        backend
            .put("a", Bytes::from("1"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("b", Bytes::from("2"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert_eq!(backend.len(), 2);
    }

    #[test]
    fn test_clear() {
        let backend = MemoryBackend::new();

        backend
            .put("a", Bytes::from("1"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("b", Bytes::from("2"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        assert_eq!(backend.len(), 2);

        backend.clear();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_with_data() {
        let mut initial = HashMap::new();
        initial.insert("key".to_string(), Bytes::from("value"));

        let backend = MemoryBackend::with_data(initial);
        assert_eq!(backend.len(), 1);
        assert!(backend
            .exists("key")
            .ok()
            .unwrap_or_else(|| panic!("Should check")));
    }

    #[test]
    fn test_clone() {
        let backend = MemoryBackend::new();
        backend
            .put("key", Bytes::from("value"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let cloned = backend;
        assert_eq!(cloned.len(), 1);
        assert_eq!(
            cloned
                .get("key")
                .ok()
                .unwrap_or_else(|| panic!("Should get")),
            Bytes::from("value")
        );
    }

    #[test]
    fn test_delete_nonexistent_is_ok() {
        let backend = MemoryBackend::new();
        let result = backend.delete("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_size_nonexistent() {
        let backend = MemoryBackend::new();
        let result = backend.size("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_default() {
        let backend = MemoryBackend::default();
        assert!(backend.is_empty());
    }

    #[test]
    fn test_debug() {
        let backend = MemoryBackend::new();
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("MemoryBackend"));
    }

    #[test]
    fn test_list_with_prefix_filter() {
        let backend = MemoryBackend::new();

        backend
            .put("data/train.parquet", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("data/test.parquet", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("metadata/info.json", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let data_keys = backend
            .list("data/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(data_keys.len(), 2);

        let metadata_keys = backend
            .list("metadata/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(metadata_keys.len(), 1);
    }
}
