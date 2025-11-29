//! Python doctest parser.
//!
//! Extracts doctests from Python source files using regex-based parsing.

use std::path::Path;

use regex::Regex;
use walkdir::WalkDir;

use super::{DocTest, DocTestCorpus};
use crate::Result;

/// Parser for extracting Python doctests from source files.
#[derive(Debug)]
pub struct DocTestParser {
    /// Regex for finding docstrings
    docstring_re: Regex,
    /// Regex for finding function/class definitions
    def_re: Regex,
}

impl Default for DocTestParser {
    fn default() -> Self {
        Self::new()
    }
}

impl DocTestParser {
    /// Create a new parser.
    ///
    /// # Panics
    /// Panics if regex compilation fails (should never happen with valid
    /// patterns).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new() -> Self {
        Self {
            // Match triple-quoted strings (""" only - most common for docstrings)
            // Can't use backreferences in Rust regex, so we match """ ... """ directly
            docstring_re: Regex::new(r#"(?s)"""(.*?)""""#).expect("valid regex"),
            // Match def/class definitions to get function names
            def_re: Regex::new(r"(?m)^(\s*)(def|class)\s+(\w+)").expect("valid regex"),
        }
    }

    /// Extract doctests from a Python source string.
    ///
    /// # Arguments
    /// * `source` - Python source code
    /// * `module` - Module name for the extracted doctests
    #[must_use]
    pub fn parse_source(&self, source: &str, module: &str) -> Vec<DocTest> {
        let mut results = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        // Build a map of line number -> (indent, name, is_class)
        let mut context_map: Vec<Option<(usize, String, bool)>> = vec![None; lines.len()];
        let mut context_stack: Vec<(usize, String, bool)> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            if let Some(caps) = self.def_re.captures(line) {
                let indent = caps.get(1).map_or(0, |m| m.as_str().len());
                let kind = caps.get(2).map_or("", |m| m.as_str());
                let name = caps.get(3).map_or("", |m| m.as_str()).to_string();
                let is_class = kind == "class";

                // Pop contexts with same or greater indent
                while context_stack
                    .last()
                    .is_some_and(|(ctx_indent, _, _)| *ctx_indent >= indent)
                {
                    context_stack.pop();
                }

                context_stack.push((indent, name, is_class));
            }

            // Store current context for this line
            context_map[i] = context_stack.last().cloned();
        }

        // Find all docstrings and extract doctests
        for caps in self.docstring_re.captures_iter(source) {
            let Some(docstring_match) = caps.get(0) else {
                continue;
            };
            let content = caps.get(1).map_or("", |m| m.as_str());

            // Find which line this docstring starts on
            let start_byte = docstring_match.start();
            let line_num = source[..start_byte].matches('\n').count();

            // Determine the function name from context
            let function_name = Self::get_function_name(line_num, &context_map, &lines);

            // Extract doctests from this docstring
            let doctests = Self::extract_from_docstring(content, module, &function_name);
            results.extend(doctests);
        }

        results
    }

    /// Get the function name for a docstring at the given line.
    fn get_function_name(
        line_num: usize,
        context_map: &[Option<(usize, String, bool)>],
        lines: &[&str],
    ) -> String {
        // Check if there's a def/class on this line or the previous line
        if line_num < lines.len() {
            // Look at current context
            if let Some((_, ref name, is_class)) = context_map.get(line_num).and_then(|c| c.clone())
            {
                // Check if we're inside a class method
                if !is_class {
                    // It's a function - check if it's inside a class
                    for i in (0..line_num).rev() {
                        if let Some((_, ref class_name, true)) =
                            context_map.get(i).and_then(|c| c.clone())
                        {
                            return format!("{class_name}.{name}");
                        }
                    }
                }
                return name.clone();
            }
        }

        // Module-level docstring
        "__module__".to_string()
    }

    /// Extract doctests from a docstring's content.
    fn extract_from_docstring(content: &str, module: &str, function: &str) -> Vec<DocTest> {
        let mut results = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Look for >>> prompt
            if let Some(input_start) = line.strip_prefix(">>>") {
                let mut input_lines = vec![format!(">>>{}", input_start)];
                i += 1;

                // Collect continuation lines (...)
                while i < lines.len() {
                    let next_line = lines[i].trim();
                    if let Some(cont) = next_line.strip_prefix("...") {
                        input_lines.push(format!("...{}", cont));
                        i += 1;
                    } else {
                        break;
                    }
                }

                // Collect expected output (until next >>> or blank line pattern)
                let mut expected_lines: Vec<&str> = Vec::new();
                // Find the base indentation of this doctest block
                let base_indent = lines
                    .get(i.saturating_sub(1))
                    .map(|l| l.len() - l.trim_start().len())
                    .unwrap_or(0);

                while i < lines.len() {
                    let next_line = lines[i];
                    let trimmed = next_line.trim();

                    // Stop at next prompt or end of doctest section
                    if trimmed.starts_with(">>>") {
                        break;
                    }

                    // Empty line might end the expected output, but tracebacks can have blank lines
                    if trimmed.is_empty() && !expected_lines.is_empty() {
                        // Peek ahead - if next non-empty line is >>> or end, we're done
                        let mut j = i + 1;
                        while j < lines.len() && lines[j].trim().is_empty() {
                            j += 1;
                        }
                        if j >= lines.len() || lines[j].trim().starts_with(">>>") {
                            break;
                        }
                        // Otherwise include the blank line (might be part of
                        // multiline output)
                    }

                    if !trimmed.is_empty() || !expected_lines.is_empty() {
                        // Strip base indentation but preserve relative indentation
                        let stripped = if next_line.len() > base_indent {
                            &next_line
                                [base_indent.min(next_line.len() - next_line.trim_start().len())..]
                        } else {
                            trimmed
                        };
                        expected_lines.push(stripped.trim_end());
                    }
                    i += 1;
                }

                // Trim trailing empty lines from expected
                while expected_lines.last().is_some_and(|l| l.is_empty()) {
                    expected_lines.pop();
                }

                let input = input_lines.join("\n");
                let expected = expected_lines.join("\n");

                results.push(DocTest::new(module, function, input, expected));
            } else {
                i += 1;
            }
        }

        results
    }

    /// Extract doctests from a Python file.
    pub fn parse_file(&self, path: &Path, module: &str) -> Result<Vec<DocTest>> {
        let source = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: Some(path.to_path_buf()),
            source: e,
        })?;
        Ok(self.parse_source(&source, module))
    }

    /// Extract doctests from a directory of Python files.
    ///
    /// # Arguments
    /// * `dir` - Directory to scan for .py files
    /// * `source` - Source identifier (e.g., "cpython")
    /// * `version` - Version string or git SHA
    pub fn parse_directory(
        &self,
        dir: &Path,
        source: &str,
        version: &str,
    ) -> Result<DocTestCorpus> {
        let mut corpus = DocTestCorpus::new(source, version);

        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "py") {
                let module = path_to_module(dir, path);
                let doctests = self.parse_file(path, &module)?;
                for dt in doctests {
                    corpus.push(dt);
                }
            }
        }

        Ok(corpus)
    }
}

/// Convert a file path to a Python module name.
///
/// Example: `/lib/python/os/path.py` with base `/lib/python` -> `os.path`
fn path_to_module(base: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(base).unwrap_or(path);
    let stem = relative.with_extension("");
    stem.to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, ".")
        .trim_end_matches(".__init__")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_module_simple() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/os.py");
        assert_eq!(path_to_module(base, path), "os");
    }

    #[test]
    fn test_path_to_module_nested() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/os/path.py");
        assert_eq!(path_to_module(base, path), "os.path");
    }

    #[test]
    fn test_path_to_module_init() {
        let base = Path::new("/lib");
        let path = Path::new("/lib/collections/__init__.py");
        assert_eq!(path_to_module(base, path), "collections");
    }

    #[test]
    fn test_extract_simple() {
        let parser = DocTestParser::new();
        let source = r#"
def foo():
    """
    >>> 1 + 1
    2
    """
    pass
"#;
        let doctests = parser.parse_source(source, "test");
        assert_eq!(doctests.len(), 1);
        assert_eq!(doctests[0].input, ">>> 1 + 1");
        assert_eq!(doctests[0].expected, "2");
    }

    #[test]
    fn test_extract_multiline_input() {
        let parser = DocTestParser::new();
        let source = r#"
def foo():
    """
    >>> x = (
    ...     1 + 2
    ... )
    >>> x
    3
    """
    pass
"#;
        let doctests = parser.parse_source(source, "test");
        assert_eq!(doctests.len(), 2);
        assert_eq!(doctests[0].input, ">>> x = (\n...     1 + 2\n... )");
        assert_eq!(doctests[0].expected, "");
    }
}
