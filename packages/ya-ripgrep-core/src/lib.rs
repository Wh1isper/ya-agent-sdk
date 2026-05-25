use globset::{GlobBuilder, GlobMatcher};
use grep_matcher::Matcher;
use grep_regex::RegexMatcher;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
struct RustGlob {
    matcher: GlobMatcher,
    recursive_prefix_matcher: Option<GlobMatcher>,
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
            if stripped.is_empty() {
                "*"
            } else {
                stripped
            }
        } else {
            normalized.as_str()
        };
        let matcher = compile_glob(glob_pattern)?;
        let recursive_prefix_matcher = glob_pattern
            .strip_prefix("**/")
            .map(compile_glob)
            .transpose()?;
        Ok(Self {
            matcher,
            recursive_prefix_matcher,
            pattern: glob_pattern.to_string(),
            anchored,
        })
    }

    fn is_match(&self, path: &str) -> bool {
        self.matches_normalized(&normalize_path(path))
    }

    fn match_many(&self, paths: Vec<String>) -> Vec<bool> {
        paths
            .iter()
            .map(|path| self.matches_normalized(&normalize_path(path)))
            .collect()
    }
}

impl RustGlob {
    fn matches_normalized(&self, normalized: &str) -> bool {
        if self.anchored && !self.pattern.contains('/') && normalized.contains('/') {
            return false;
        }
        if self.pattern == "**" || self.pattern == "**/*" {
            return true;
        }
        if self.matcher.is_match(normalized) {
            return true;
        }
        if let Some(matcher) = &self.recursive_prefix_matcher {
            if matcher.is_match(normalized) {
                return true;
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

    fn search_bytes(
        &self,
        data: &[u8],
        context_lines: usize,
        max_matches: isize,
    ) -> PyResult<Vec<(usize, String, String, usize)>> {
        search_bytes_with_matcher(&self.matcher, data, context_lines, max_matches)
    }
}

#[pyfunction]
fn match_glob(path: &str, pattern: &str) -> PyResult<bool> {
    Ok(RustGlob::new(pattern)?.is_match(path))
}

#[pyfunction]
fn match_globs(paths: Vec<String>, pattern: &str) -> PyResult<Vec<bool>> {
    Ok(RustGlob::new(pattern)?.match_many(paths))
}

#[pyfunction]
fn regex_is_match(pattern: &str, text: &str) -> PyResult<bool> {
    RustRegex::new(pattern)?.is_match(text)
}

#[pyfunction]
fn regex_search_bytes(
    pattern: &str,
    data: &[u8],
    context_lines: usize,
    max_matches: isize,
) -> PyResult<Vec<(usize, String, String, usize)>> {
    let matcher = RegexMatcher::new_line_matcher(pattern)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    search_bytes_with_matcher(&matcher, data, context_lines, max_matches)
}

fn compile_glob(pattern: &str) -> PyResult<GlobMatcher> {
    GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .map_err(|err| PyValueError::new_err(err.to_string()))
        .map(|glob| glob.compile_matcher())
}

fn search_bytes_with_matcher(
    matcher: &RegexMatcher,
    data: &[u8],
    context_lines: usize,
    max_matches: isize,
) -> PyResult<Vec<(usize, String, String, usize)>> {
    let text = String::from_utf8_lossy(data);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut matches = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        if max_matches > 0 && matches.len() >= max_matches as usize {
            break;
        }
        if matcher
            .is_match(line.as_bytes())
            .map_err(|err| PyValueError::new_err(err.to_string()))?
        {
            let line_number = index + 1;
            let start_index = index.saturating_sub(context_lines);
            let end_index = (index + context_lines + 1).min(lines.len());
            let context = lines[start_index..end_index].concat();
            let context_start_line = start_index + 1;
            let matching_line = line.trim_end_matches('\n').to_string();
            matches.push((line_number, matching_line, context, context_start_line));
        }
    }

    Ok(matches)
}

fn normalize_path(path: &str) -> String {
    let mut normalized = path.replace('\\', "/");
    if let Some(stripped) = normalized.strip_prefix("./") {
        normalized = stripped.to_string();
    }
    if normalized.is_empty() {
        ".".to_string()
    } else {
        normalized
    }
}

#[pymodule]
fn ya_ripgrep_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustGlob>()?;
    m.add_class::<RustRegex>()?;
    m.add_function(wrap_pyfunction!(match_glob, m)?)?;
    m.add_function(wrap_pyfunction!(match_globs, m)?)?;
    m.add_function(wrap_pyfunction!(regex_is_match, m)?)?;
    m.add_function(wrap_pyfunction!(regex_search_bytes, m)?)?;
    Ok(())
}
