use hyperstreamdb::Table;
use arrow::array::{
    Int32Array, BooleanArray, Time32MillisecondArray, BinaryArray, 
    Decimal128Array, StringArray, DictionaryArray, Int8Array
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_all_types_indexing() -> anyhow::Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let temp_dir = tempfile::tempdir()?;
        let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
        
        // 1. Create Table
        let mut table = Table::new_async(uri.clone()).await?;
        
        // 2. Schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("flag", DataType::Boolean, false),
            Field::new("time", DataType::Time32(TimeUnit::Millisecond), false),
            Field::new("blob", DataType::Binary, false),
            Field::new("money", DataType::Decimal128(10, 2), false), // Precision 10, Scale 2
            Field::new("category", DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)), false),
        ]));

        // 3. Data
        let ids = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let flags = BooleanArray::from(vec![true, false, true, false, true]);
        let times = Time32MillisecondArray::from(vec![1000, 2000, 3000, 4000, 5000]); // 1s, 2s...
        let blobs = BinaryArray::from_iter_values(vec![
            &b"foo"[..], &b"bar"[..], &b"baz"[..], &b"qux"[..], &b"quux"[..]
        ]);
        
        // Decimal: 1.11, 2.22, 3.33, 4.44, 5.55 -> stored as 111, 222...
        let decimals = Decimal128Array::from_iter_values(vec![111, 222, 333, 444, 555])
            .with_precision_and_scale(10, 2)?;

        // Dictionary
        let keys = Int8Array::from(vec![0, 1, 0, 1, 2]);
        let values = StringArray::from(vec!["cat", "dog", "bird"]); // bird (2) is used
        let dict = DictionaryArray::try_new(keys, Arc::new(values))?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(flags),
                Arc::new(times),
                Arc::new(blobs),
                Arc::new(decimals),
                Arc::new(dict),
            ]
        )?;

        // 4. Write & Index
        table.index_all_columns_async().await?; // Enable indexing for everything
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;

        // 5. Verification Queries

        // A. Boolean Index: flag = true (should be 1, 3, 5)
        let res = table.read_async(Some("flag = true"), None, None).await?;
        assert_eq!(res.iter().map(|b| b.num_rows()).sum::<usize>(), 3);

        // B. Time Index: time > 2500 (should be 3, 4, 5)
        // Note: Filter parser needs to support time literals or we pass explicit integer?
        // HyperStream filter currently takes just simple string. Planner parses it.
        // If we say "time > 2500" (integer), it should work against Time32 keys (which are i32).
        // Update: DataFusion is strict about Int32 > Int64. We cast time to bigint.
        let res = table.read_async(Some("cast(time as bigint) > 2500"), None, None).await?;
        assert_eq!(res.iter().map(|b| b.num_rows()).sum::<usize>(), 3);

        // C. Binary Index: blob = 'foo'
        // DataFusion issues Utf8View vs Utf8. Cast both to string.
        let res = table.read_async(Some("cast(blob as string) = cast('foo' as string)"), None, None).await?;
        // 'foo' is row 1
        assert_eq!(res.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

        // D. Decimal Index: money > 3.00
        // DataFusion Decimal > Float requires cast.
        let res = table.read_async(Some("cast(money as double) > 3.0"), None, None).await?;
        assert_eq!(res.iter().map(|b| b.num_rows()).sum::<usize>(), 3);

        // E. Dictionary Index: unpacks to "cat", "dog", "bird"
        // Row 1: cat, Row 2: dog, Row 3: cat, Row 4: dog, Row 5: bird
        // Query: category = 'cat' -> Row 1, 3
        let res = table.read_async(Some("category = 'cat'"), None, None).await?;
        assert_eq!(res.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

        Ok(())
    })
}
