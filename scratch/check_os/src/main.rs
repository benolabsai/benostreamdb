use object_store::{local::LocalFileSystem, ObjectStore, ObjectStoreExt, PutMode, PutOptions, path::Path};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let store = Arc::new(LocalFileSystem::new_with_prefix(tmp.path())?) as Arc<dyn ObjectStore>;
    let path = Path::from("test.json");
    
    store.put(&path, "v1".into()).await?;
    
    let opts = PutOptions {
        mode: PutMode::Create,
        ..Default::default()
    };
    
    let res = store.put_opts(&path, "v2".into(), opts).await;
    match res {
        Err(e) => {
            println!("Error: {:?}", e);
            println!("Display: {}", e);
            println!("Contains 'already exists': {}", e.to_string().contains("already exists"));
        }
        Ok(_) => println!("Successfully overwrote (unexpected with PutMode::Create)"),
    }
    
    Ok(())
}
