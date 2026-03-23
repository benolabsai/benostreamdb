use hyperstreamdb::core::puffin::{PuffinWriter, PuffinReader};
use std::io::Cursor;
use std::collections::HashMap;

#[test]
fn test_puffin_write_read() -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = Cursor::new(Vec::new());
    
    let mut writer = PuffinWriter::new(&mut buffer).unwrap();
    writer.add_blob(
        "test-blob".to_string(),
        vec![1],
        100,
        1,
        b"Hello Puffin!",
        HashMap::new()
    )?;
    
    writer.add_blob(
        "another-blob".to_string(),
        vec![2, 3],
        100,
        1,
        b"Binary Data",
        HashMap::new()
    )?;
    
    writer.finish().unwrap();
    
    let bytes = buffer.into_inner();
    let mut reader = PuffinReader::new(Cursor::new(bytes)).unwrap();
    
    assert_eq!(reader.footer().blobs.len(), 2);
    
    let data1 = reader.read_blob(0).unwrap();
    assert_eq!(data1, b"Hello Puffin!");
    assert_eq!(reader.footer().blobs[0].r#type, "test-blob");
    assert_eq!(reader.footer().blobs[0].fields, vec![1]);
    
    let data2 = reader.read_blob(1).unwrap();
    assert_eq!(data2, b"Binary Data");
    assert_eq!(reader.footer().blobs[1].r#type, "another-blob");
    
    Ok(())
}
