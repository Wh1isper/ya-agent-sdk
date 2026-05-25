use globset::{GlobBuilder, GlobMatcher};
use grep_matcher::Matcher;
use grep_regex::RegexMatcher;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
struct RustGlob {
    matcher: GlobMatcher,
    pattern: String,
    anchored: bool,
}

#[pymethods]
impl RustGlob {
    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        let mut normalized = pattern.replace('\\', "/");
        if normalized.is_empty() {
            normalized = "**/*".to_string();
        }
        if let Some(stripped) = normalized.strip_prefix("./") {
            normalized = stripped.to_string();
        }
        let anchored = normalized.starts_with('/');
        let glob_pattern = if anchored {
            let stripped = normalized.trim_start_matches('/');
            if stripped.is_empty() { "*" } else { stripped }
        } else {
            normalized.as_str()
        };
        let matcher = GlobBuilder::new(glob_pattern)
            .literal_separator(false)
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .compile_matcher();
        Ok(Self { matcher, pattern: glob_pattern.to_string(), anchored })
    }

    fn is_match(&self, path: &str) -> bool {
        let normalized = normalize_path(path);
        if self.anchored && !self.pattern.contains('/') && normalized.contains('/') {
            return false;
        }
        if self.pattern == "**" || self.pattern == "**/*" {
            return true;
        }
        if self.matcher.is_match(&normalized) {
            return true;
        }
        if let Some(without_prefix) = self.pattern.strip_prefix("**/") {
            if let Ok(glob) = GlobBuilder::new(without_prefix).literal_separator(false).build() {
                if glob.compile_matcher().is_match(&normalized) {
                    return true;
                }
            }
        }
        if !self.anchored && !self.pattern.contains('/') {
            if let Some(name) = normalized.rsplit('/').next() {
                return self.matcher.is_match(name);
            }
        }
        false
    }
}

#[pyclass]
struct RustRegex {
    matcher: RegexMatcher,
}

#[pymethods]
impl RustRegex {
    #[new]
    fn new(pattern: &str) -> PyResult<Self> {
        let matcher = RegexMatcher::new_line_matcher(pattern)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { matcher })
    }

    fn is_match(&self, text: &str) -> PyResult<bool> {
        self.matcher
            .is_match(text.as_bytes())
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pyfunction]
fn match_glob(path: &str, pattern: &str) -> PyResult<bool> {
    Ok(RustGlob::new(pattern)?.is_match(path))
}

#[pyfunction]
fn regex_is_match(pattern: &str, text: &str) -> PyResult<bool> {
    RustRegex::new(pattern)?.is_match(text)
}

fn normalize_path(path: &str) -> String {
    let mut normalized = path.replace('\\', "/");
    if let Some(stripped) = normalized.strip_prefix("./") {
        normalized = stripped.to_string();
    }
    if normalized.is_empty() { ".".to_string() } else { normalized }
}

#[pymodule]
fn ya_ripgrep_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustGlob>()?;
    m.add_class::<RustRegex>()?;
    m.add_function(wrap_pyfunction!(match_glob, m)?)?;
    m.add_function(wrap_pyfunction!(regex_is_match, m)?)?;
    Ok(())
}
