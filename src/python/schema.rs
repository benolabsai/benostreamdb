// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;
use std::collections::HashMap;

#[pyclass(name = "DataType")]
#[derive(Clone, Debug)]
pub struct PyDataType {
    pub(crate) dt: DataType,
}

#[pymethods]
impl PyDataType {
    #[staticmethod]
    fn int8() -> Self { Self { dt: DataType::Int8 } }
    #[staticmethod]
    fn int16() -> Self { Self { dt: DataType::Int16 } }
    #[staticmethod]
    fn int32() -> Self { Self { dt: DataType::Int32 } }
    #[staticmethod]
    fn int64() -> Self { Self { dt: DataType::Int64 } }
    #[staticmethod]
    fn uint8() -> Self { Self { dt: DataType::UInt8 } }
    #[staticmethod]
    fn uint16() -> Self { Self { dt: DataType::UInt16 } }
    #[staticmethod]
    fn uint32() -> Self { Self { dt: DataType::UInt32 } }
    #[staticmethod]
    fn uint64() -> Self { Self { dt: DataType::UInt64 } }
    #[staticmethod]
    fn float16() -> Self { Self { dt: DataType::Float16 } }
    #[staticmethod]
    fn float32() -> Self { Self { dt: DataType::Float32 } }
    #[staticmethod]
    fn float64() -> Self { Self { dt: DataType::Float64 } }
    #[staticmethod]
    fn string() -> Self { Self { dt: DataType::Utf8 } }
    #[staticmethod]
    fn binary() -> Self { Self { dt: DataType::Binary } }
    #[staticmethod]
    fn boolean() -> Self { Self { dt: DataType::Boolean } }
    #[staticmethod]
    fn date32() -> Self { Self { dt: DataType::Date32 } }
    #[staticmethod]
    fn date64() -> Self { Self { dt: DataType::Date64 } }
    #[staticmethod]
    fn timestamp_ms() -> Self { Self { dt: DataType::Timestamp(TimeUnit::Millisecond, None) } }
    #[staticmethod]
    fn timestamp_us() -> Self { Self { dt: DataType::Timestamp(TimeUnit::Microsecond, None) } }
    #[staticmethod]
    #[pyo3(signature = (dim, nullable=true))]
    fn vector(dim: usize, nullable: bool) -> Self {
        Self {
            dt: DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, nullable)),
                dim as i32,
            )
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.dt)
    }
}

#[pyclass(name = "Field")]
#[derive(Clone, Debug)]
pub struct PyField {
    pub(crate) inner: Field,
}

#[pymethods]
impl PyField {
    #[new]
    #[pyo3(signature = (name, data_type, nullable=true, metadata=None))]
    fn new(name: String, data_type: PyDataType, nullable: bool, metadata: Option<HashMap<String, String>>) -> Self {
        let mut field = Field::new(name, data_type.dt, nullable);
        if let Some(m) = metadata {
            field = field.with_metadata(m);
        }
        Self { inner: field }
    }

    fn __repr__(&self) -> String {
        format!("Field(name={}, type={:?}, nullable={})", self.inner.name(), self.inner.data_type(), self.inner.is_nullable())
    }
}

#[pyclass(name = "PartitionField")]
#[derive(Clone, Debug)]
pub struct PyPartitionField {
    #[pyo3(get, set)]
    pub source_ids: Vec<i32>,
    #[pyo3(get, set)]
    pub field_id: Option<i32>,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub transform: String,
}

#[pymethods]
impl PyPartitionField {
    #[new]
    #[pyo3(signature = (source_ids, name, transform, field_id=None))]
    fn new(source_ids: Vec<i32>, name: String, transform: String, field_id: Option<i32>) -> Self {
        Self { source_ids, field_id, name, transform }
    }
}

#[pyclass(name = "Schema")]
#[derive(Clone, Debug)]
pub struct PySchema {
    pub(crate) inner: arrow::datatypes::SchemaRef,
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (fields, metadata=None))]
    fn new(fields: Vec<PyField>, metadata: Option<HashMap<String, String>>) -> Self {
        let arrow_fields: Vec<Field> = fields.into_iter().map(|f| f.inner).collect();
        let mut schema = Schema::new(arrow_fields);
        if let Some(m) = metadata {
            schema = schema.with_metadata(m);
        }
        Self {
            inner: Arc::new(schema),
        }
    }

    fn __repr__(&self) -> String {
        format!("Schema(fields={:?})", self.inner.fields())
    }
}
