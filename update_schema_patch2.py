
import os

path = "src/core/sql/session.rs"
with open(path, "r") as f:
    text = f.read()
text = text.replace("let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().into());", "let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().as_arrow().clone());")
# wait df.schema() might return DFSchema, wait, let us just use `std::sync::Arc::new(arrow::datatypes::Schema::from(df.schema()))`
with open(path, "w") as f:
    f.write(text)

path2 = "src/python_binding.rs"
with open(path2, "r") as f:
    text2 = f.read()
text2 = text2.replace("let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().into());", "let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(arrow::datatypes::Schema::from(df.schema()));")
with open(path2, "w") as f:
    f.write(text2)
