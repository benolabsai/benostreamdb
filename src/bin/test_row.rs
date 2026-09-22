use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use std::sync::Arc;

fn main() {
    let id_array = Int32Array::from(vec![1, 2, 3]);
    let name_array = StringArray::from(vec!["a", "b", "c"]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let fields = vec![
        SortField::new(DataType::Int32),
        SortField::new(DataType::Utf8),
    ];

    let converter = RowConverter::new(fields).unwrap();
    let rows = converter.convert_columns(batch.columns()).unwrap();

    for row in rows.iter() {
        println!("{:?}", row.as_ref());
    }
}
