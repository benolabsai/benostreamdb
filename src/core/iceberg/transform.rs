// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::Array;

#[derive(Debug, Clone, PartialEq)]
pub enum IcebergTransform {
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
    Void,
}

impl IcebergTransform {
    pub fn parse(s: &str) -> Self {
        match s {
            "identity" => Self::Identity,
            "year" => Self::Year,
            "month" => Self::Month,
            "day" => Self::Day,
            "hour" => Self::Hour,
            "void" => Self::Void,
            _ if s.starts_with("bucket[") => {
                let n = s
                    .trim_start_matches("bucket[")
                    .trim_end_matches(']')
                    .parse()
                    .unwrap_or(0);
                Self::Bucket(n)
            }
            _ if s.starts_with("truncate[") => {
                let n = s
                    .trim_start_matches("truncate[")
                    .trim_end_matches(']')
                    .parse()
                    .unwrap_or(0);
                Self::Truncate(n)
            }
            _ => Self::Void,
        }
    }

    pub fn apply(&self, array: &dyn Array, row_i: usize) -> serde_json::Value {
        self.apply_multi(&[array], row_i)
    }

    pub fn apply_multi(&self, arrays: &[&dyn Array], row_i: usize) -> serde_json::Value {
        if arrays.is_empty() {
            return serde_json::Value::Null;
        }
        if arrays.iter().any(|a| a.is_null(row_i)) {
            return serde_json::Value::Null;
        }

        use chrono::Datelike;

        match self {
            Self::Identity => {
                let array = arrays[0];
                match array.data_type() {
                    arrow::datatypes::DataType::Int32 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Int32Array>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        serde_json::json!(a.value(row_i))
                    }
                    arrow::datatypes::DataType::Int64 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Int64Array>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        serde_json::json!(a.value(row_i))
                    }
                    arrow::datatypes::DataType::Utf8 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::StringArray>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        serde_json::json!(a.value(row_i))
                    }
                    arrow::datatypes::DataType::Date32 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Date32Array>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        serde_json::json!(a.value(row_i))
                    }
                    arrow::datatypes::DataType::Timestamp(_, _) => {
                        let a = if let Some(arr) = array
                            .as_any()
                            .downcast_ref::<arrow::array::TimestampMicrosecondArray>(
                        ) {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        serde_json::json!(a.value(row_i))
                    }
                    _ => serde_json::Value::Null,
                }
            }
            Self::Bucket(n) => {
                let mut hash_val: u32 = 0;
                for array in arrays {
                    let field_hash = match array.data_type() {
                        arrow::datatypes::DataType::Int32 => {
                            let a = if let Some(arr) =
                                array.as_any().downcast_ref::<arrow::array::Int32Array>()
                            {
                                arr
                            } else {
                                return serde_json::Value::Null;
                            };
                            let val = a.value(row_i);
                            let bytes = (val as i64).to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        }
                        arrow::datatypes::DataType::Int64 => {
                            let a = if let Some(arr) =
                                array.as_any().downcast_ref::<arrow::array::Int64Array>()
                            {
                                arr
                            } else {
                                return serde_json::Value::Null;
                            };
                            let val = a.value(row_i);
                            let bytes = val.to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        }
                        arrow::datatypes::DataType::Utf8 => {
                            let a = if let Some(arr) =
                                array.as_any().downcast_ref::<arrow::array::StringArray>()
                            {
                                arr
                            } else {
                                return serde_json::Value::Null;
                            };
                            let s = a.value(row_i);
                            murmur3_32_x86(s.as_bytes(), hash_val)
                        }
                        arrow::datatypes::DataType::Date32 => {
                            let a = if let Some(arr) =
                                array.as_any().downcast_ref::<arrow::array::Date32Array>()
                            {
                                arr
                            } else {
                                return serde_json::Value::Null;
                            };
                            let val = a.value(row_i);
                            let bytes = (val as i64).to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        }
                        _ => 0,
                    };
                    hash_val = field_hash;
                }
                // Iceberg Bucketing: (hash & Integer.MAX_VALUE) % N
                serde_json::json!((hash_val & 0x7FFFFFFF) % *n)
            }
            Self::Truncate(w) => {
                let array = arrays[0];
                match array.data_type() {
                    arrow::datatypes::DataType::Utf8 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::StringArray>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        let s = a.value(row_i);
                        let limit = (*w as usize).min(s.len());
                        serde_json::json!(&s[..limit])
                    }
                    arrow::datatypes::DataType::Int32 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Int32Array>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        let v = a.value(row_i);
                        serde_json::json!(v - (v % (*w as i32)))
                    }
                    arrow::datatypes::DataType::Int64 => {
                        let a = if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Int64Array>()
                        {
                            arr
                        } else {
                            return serde_json::Value::Null;
                        };
                        let v = a.value(row_i);
                        serde_json::json!(v - (v % (*w as i64)))
                    }
                    _ => serde_json::Value::Null,
                }
            }
            Self::Year => {
                let array = arrays[0];
                // Years from 1970
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = if let Some(arr) =
                        array.as_any().downcast_ref::<arrow::array::Date32Array>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let days = a.value(row_i);
                    // 1970-01-01 is epoch.
                    let opt_date = chrono::NaiveDate::from_num_days_from_ce_opt(days + 719163);
                    if let Some(d) = opt_date {
                        serde_json::json!(d.year() - 1970)
                    } else {
                        serde_json::Value::Null
                    }
                } else if let arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    _,
                ) = array.data_type()
                {
                    let a = if let Some(arr) = array
                        .as_any()
                        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let micros = a.value(row_i);
                    let seconds = micros / 1_000_000;
                    let opt_dt = chrono::DateTime::from_timestamp(seconds, 0);
                    if let Some(dt) = opt_dt {
                        serde_json::json!(dt.year() - 1970)
                    } else {
                        serde_json::Value::Null
                    }
                } else {
                    serde_json::Value::Null
                }
            }
            Self::Month => {
                let array = arrays[0];
                // Months from 1970-01-01
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = if let Some(arr) =
                        array.as_any().downcast_ref::<arrow::array::Date32Array>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let days = a.value(row_i);
                    let opt_date = chrono::NaiveDate::from_num_days_from_ce_opt(days + 719163);
                    if let Some(d) = opt_date {
                        serde_json::json!((d.year() - 1970) * 12 + (d.month() as i32) - 1)
                    } else {
                        serde_json::Value::Null
                    }
                } else if let arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    _,
                ) = array.data_type()
                {
                    let a = if let Some(arr) = array
                        .as_any()
                        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let micros = a.value(row_i);
                    let seconds = micros / 1_000_000;
                    let opt_dt = chrono::DateTime::from_timestamp(seconds, 0);
                    if let Some(dt) = opt_dt {
                        serde_json::json!((dt.year() - 1970) * 12 + (dt.month() as i32) - 1)
                    } else {
                        serde_json::Value::Null
                    }
                } else {
                    serde_json::Value::Null
                }
            }
            Self::Day => {
                let array = arrays[0];
                // Days from 1970-01-01
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = if let Some(arr) =
                        array.as_any().downcast_ref::<arrow::array::Date32Array>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    serde_json::json!(a.value(row_i))
                } else if let arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    _,
                ) = array.data_type()
                {
                    let a = if let Some(arr) = array
                        .as_any()
                        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let micros = a.value(row_i);
                    // Spec: input timestamp (micros) -> days from epoch
                    serde_json::json!(micros / (1_000_000 * 60 * 60 * 24))
                } else {
                    serde_json::Value::Null
                }
            }
            Self::Hour => {
                let array = arrays[0];
                // Hours from 1970-01-01 00:00:00
                if let arrow::datatypes::DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    _,
                ) = array.data_type()
                {
                    let a = if let Some(arr) = array
                        .as_any()
                        .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    {
                        arr
                    } else {
                        return serde_json::Value::Null;
                    };
                    let micros = a.value(row_i);
                    serde_json::json!(micros / (1_000_000 * 60 * 60))
                } else {
                    serde_json::Value::Null
                }
            }
            Self::Void => serde_json::Value::Null,
        }
    }
}

/// Wrapper around murmur3 crate to ensure x86 32-bit implementation
/// Aligned with official `iceberg-rust` implementation.
pub fn murmur3_32_x86(data: &[u8], seed: u32) -> u32 {
    murmur3::murmur3_32(&mut std::io::Cursor::new(data), seed).unwrap_or(0)
}
