use pyo3::prelude::*;
use pyo3::types::PyDict;

pub struct PythonCallbackFollowUpGenerator {
    pub callback: Py<PyAny>,
}

impl crate::core::sql::graph_udf::drift_search::DriftFollowUpGenerator
    for PythonCallbackFollowUpGenerator
{
    fn generate_primer(
        &self,
        query: &str,
        top_communities: &[u64],
    ) -> Vec<(String, f64, Vec<u64>)> {
        #[allow(deprecated)]
        let attempted = Python::with_gil(|py| -> PyResult<Vec<(String, f64, Vec<u64>)>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("query", query)?;
            kwargs.set_item("top_communities", top_communities)?;
            kwargs.set_item("phase", "primer")?;

            let result = self.callback.bind(py).call((), Some(&kwargs))?;
            Ok(result.extract().unwrap_or_default())
        });
        // The trait returns a plain Vec, so a callback failure degrades to "no
        // suggestions" with a log rather than panicking the search.
        attempted.unwrap_or_else(|e| {
            tracing::error!(error = %e, "drift primer callback failed; continuing without suggestions");
            Vec::new()
        })
    }

    fn generate_follow_ups(
        &self,
        query: &str,
        discovered_nodes: &[u64],
        round_num: u32,
    ) -> Vec<(String, f64, Vec<u64>)> {
        #[allow(deprecated)]
        let attempted = Python::with_gil(|py| -> PyResult<Vec<(String, f64, Vec<u64>)>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("query", query)?;
            kwargs.set_item("discovered_nodes", discovered_nodes)?;
            kwargs.set_item("round_num", round_num)?;
            kwargs.set_item("phase", "follow_up")?;

            let result = self.callback.bind(py).call((), Some(&kwargs))?;
            Ok(result.extract().unwrap_or_default())
        });
        attempted.unwrap_or_else(|e| {
            tracing::error!(error = %e, "drift follow-up callback failed; continuing without suggestions");
            Vec::new()
        })
    }
}
