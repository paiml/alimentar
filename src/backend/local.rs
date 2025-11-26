//! Local filesystem storage backend.

use std::{
    fs,
    path::{Path, PathBuf},
};

use bytes::Bytes;

use super::StorageBackend;
use crate::error::{Error, Result};

/// A storage backend using the local filesystem.
///
/// All keys are relative to the configured root directory.
///
/// # Example
///
/// ```no_run
/// use alimentar::backend::LocalBackend;
///
/// let backend = LocalBackend::new("/data/datasets").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LocalBackend {
    root: PathBuf,
}

impl LocalBackend {
    /// Creates a new local backend with the given root directory.
    ///
    /// Creates the directory if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| Error::io(e, &root))?;
        Ok(Self { root })
    }

    /// Returns the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves a key to a full filesystem path.
    fn resolve_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl StorageBackend for LocalBackend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let search_path = self.resolve_path(prefix);

        // If the prefix is a directory, list its contents
        // If it's a file prefix, list matching files in parent
        let (dir_to_search, file_prefix) = if search_path.is_dir() {
            (search_path, String::new())
        } else {
            let parent = search_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| self.root.clone());
            let prefix_name = search_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            (parent, prefix_name)
        };

        if !dir_to_search.exists() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        self.list_recursive(&dir_to_search, &file_prefix, &mut results)?;
        Ok(results)
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        let path = self.resolve_path(key);
        let data = fs::read(&path).map_err(|e| Error::io(e, &path))?;
        Ok(Bytes::from(data))
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        let path = self.resolve_path(key);

        // Create parent directories
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(e, parent))?;
        }

        fs::write(&path, &data).map_err(|e| Error::io(e, &path))
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.resolve_path(key);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| Error::io(e, &path))?;
        }
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let path = self.resolve_path(key);
        Ok(path.exists())
    }

    fn size(&self, key: &str) -> Result<u64> {
        let path = self.resolve_path(key);
        let metadata = fs::metadata(&path).map_err(|e| Error::io(e, &path))?;
        Ok(metadata.len())
    }
}

impl LocalBackend {
    /// Recursively lists files, collecting paths relative to root.
    fn list_recursive(&self, dir: &Path, prefix: &str, results: &mut Vec<String>) -> Result<()> {
        let entries = fs::read_dir(dir).map_err(|e| Error::io(e, dir))?;

        for entry in entries {
            let entry = entry.map_err(|e| Error::io(e, dir))?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            // Check prefix filter
            if !prefix.is_empty() && !file_name.starts_with(prefix) {
                continue;
            }

            if path.is_file() {
                // Get path relative to root
                if let Ok(relative) = path.strip_prefix(&self.root) {
                    results.push(relative.to_string_lossy().to_string());
                }
            } else if path.is_dir() {
                // Recurse into subdirectory (no prefix filter for subdirs)
                self.list_recursive(&path, "", results)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_backend() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path());
        assert!(backend.is_ok());
    }

    #[test]
    fn test_put_and_get() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let data = Bytes::from("hello world");
        backend
            .put("test.txt", data.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should put data"));

        let retrieved = backend
            .get("test.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should get data"));
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_put_creates_directories() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let data = Bytes::from("nested data");
        backend
            .put("a/b/c/test.txt", data)
            .ok()
            .unwrap_or_else(|| panic!("Should put nested data"));

        assert!(backend
            .exists("a/b/c/test.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists")));
    }

    #[test]
    fn test_exists() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        assert!(!backend
            .exists("nonexistent.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists")));

        backend
            .put("exists.txt", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put data"));

        assert!(backend
            .exists("exists.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists")));
    }

    #[test]
    fn test_delete() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        backend
            .put("to_delete.txt", Bytes::from("data"))
            .ok()
            .unwrap_or_else(|| panic!("Should put data"));

        assert!(backend
            .exists("to_delete.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should exist")));

        backend
            .delete("to_delete.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should delete"));

        assert!(!backend
            .exists("to_delete.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should not exist")));
    }

    #[test]
    fn test_delete_nonexistent_is_ok() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        // Deleting a non-existent file should not error
        let result = backend.delete("does_not_exist.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_size() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let data = Bytes::from("12345678901234567890"); // 20 bytes
        backend
            .put("sized.txt", data)
            .ok()
            .unwrap_or_else(|| panic!("Should put data"));

        let size = backend
            .size("sized.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should get size"));
        assert_eq!(size, 20);
    }

    #[test]
    fn test_list_empty() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let files = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert!(files.is_empty());
    }

    #[test]
    fn test_list_files() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        backend
            .put("file1.txt", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("file2.txt", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("other.txt", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let all_files = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(all_files.len(), 3);

        let file_files = backend
            .list("file")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(file_files.len(), 2);
    }

    #[test]
    fn test_list_nested() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        backend
            .put("dir1/file1.txt", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("dir1/file2.txt", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("dir2/file3.txt", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let all_files = backend
            .list("")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(all_files.len(), 3);

        let dir1_files = backend
            .list("dir1")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(dir1_files.len(), 2);
    }

    #[test]
    fn test_root() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        assert_eq!(backend.root(), temp_dir.path());
    }

    #[test]
    fn test_get_nonexistent_error() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let result = backend.get("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_size_nonexistent_error() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let result = backend.size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_debug() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("LocalBackend"));
    }

    #[test]
    fn test_clone() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let cloned = backend.clone();
        assert_eq!(cloned.root(), backend.root());
    }

    #[test]
    fn test_list_nonexistent_prefix() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        let result = backend
            .list("nonexistent/path/")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_list_with_file_prefix() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        backend
            .put("train_data.parquet", Bytes::from("a"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("train_labels.parquet", Bytes::from("b"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));
        backend
            .put("test_data.parquet", Bytes::from("c"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        // List with file prefix
        let train_files = backend
            .list("train")
            .ok()
            .unwrap_or_else(|| panic!("Should list"));
        assert_eq!(train_files.len(), 2);
    }

    #[test]
    fn test_deeply_nested_structure() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let backend = LocalBackend::new(temp_dir.path())
            .ok()
            .unwrap_or_else(|| panic!("Should create backend"));

        backend
            .put("a/b/c/d/e/f/deep.txt", Bytes::from("deep content"))
            .ok()
            .unwrap_or_else(|| panic!("Should put"));

        let exists = backend
            .exists("a/b/c/d/e/f/deep.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should check exists"));
        assert!(exists);

        let content = backend
            .get("a/b/c/d/e/f/deep.txt")
            .ok()
            .unwrap_or_else(|| panic!("Should get"));
        assert_eq!(content, Bytes::from("deep content"));
    }
}
