// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Python binding for `Table::subscribe()` — a live change feed.
//!
//! ```python
//! sub = table.subscribe()                 # or table.subscribe_filtered("age > 30")
//! ev = sub.recv(timeout_ms=1000)          # {"event_type": "batch", "rows": 3, "data": <pyarrow.Table>}
//! ev = sub.try_recv()                     # non-blocking; None when idle
//! ```

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Runtime;

use super::helpers::arrow_batches_to_pyarrow;
use crate::core::table::{Subscription, TableEvent};

/// A live subscription to a table's committed changes.
#[pyclass]
pub struct PySubscription {
    sub: Subscription,
    rt: Arc<Runtime>,
}

impl PySubscription {
    pub(crate) fn new(sub: Subscription, rt: Arc<Runtime>) -> Self {
        Self { sub, rt }
    }
}

#[pymethods]
impl PySubscription {
    /// Block for the next event. `timeout_ms=None` waits indefinitely; a
    /// timeout returns `None`.
    #[pyo3(signature = (timeout_ms=None))]
    fn recv(&mut self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<Option<Py<PyAny>>> {
        let rt = self.rt.clone();
        let sub = &mut self.sub;
        let res = py
            .detach(|| {
                rt.block_on(async {
                    match timeout_ms {
                        Some(ms) => {
                            match tokio::time::timeout(Duration::from_millis(ms), sub.recv()).await
                            {
                                Ok(r) => r.map(Some),
                                Err(_) => Ok(None),
                            }
                        }
                        None => sub.recv().await.map(Some),
                    }
                })
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        match res {
            Some(ev) => Ok(Some(event_to_py(py, ev)?)),
            None => Ok(None),
        }
    }

    /// Non-blocking variant of `recv`; returns `None` when no event is ready.
    fn try_recv(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self
            .sub
            .try_recv()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        {
            Some(ev) => Ok(Some(event_to_py(py, ev)?)),
            None => Ok(None),
        }
    }

    /// Unsubscribe now (idempotent). Dropping the object also unsubscribes.
    fn close(&mut self) {
        self.sub.close();
    }

    /// Whether this subscription has been closed.
    fn is_closed(&self) -> bool {
        self.sub.is_closed()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _ty: Option<&Bound<'_, PyAny>>,
        _val: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.sub.close();
        false
    }
}

/// Convert a [`TableEvent`] into a Python dict.
fn event_to_py(py: Python<'_>, ev: TableEvent) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match ev {
        TableEvent::Batch(batch) => {
            d.set_item("event_type", "batch")?;
            d.set_item("rows", batch.num_rows())?;
            let schema = batch.schema();
            let table = arrow_batches_to_pyarrow(py, vec![batch], schema)?;
            d.set_item("data", table)?;
        }
        TableEvent::Commit { rows } => {
            d.set_item("event_type", "commit")?;
            d.set_item("rows", rows)?;
            d.set_item("data", py.None())?;
        }
    }
    Ok(d.into_any().unbind())
}
