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
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("query", query).unwrap();
            kwargs.set_item("top_communities", top_communities).unwrap();
            kwargs.set_item("phase", "primer").unwrap();

            let result = self.callback.bind(py).call((), Some(&kwargs)).unwrap();
            let list: Vec<(String, f64, Vec<u64>)> = result.extract().unwrap_or_default();
            list
        })
    }

    fn generate_follow_ups(
        &self,
        query: &str,
        discovered_nodes: &[u64],
        round_num: u32,
    ) -> Vec<(String, f64, Vec<u64>)> {
        #[allow(deprecated)]
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("query", query).unwrap();
            kwargs
                .set_item("discovered_nodes", discovered_nodes)
                .unwrap();
            kwargs.set_item("round_num", round_num).unwrap();
            kwargs.set_item("phase", "follow_up").unwrap();

            let result = self.callback.bind(py).call((), Some(&kwargs)).unwrap();
            let list: Vec<(String, f64, Vec<u64>)> = result.extract().unwrap_or_default();
            list
        })
    }
}
