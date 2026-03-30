// Copyright (c) 2026 Richard Albright. All rights reserved.

use object_store::{aws::AmazonS3Builder, azure::MicrosoftAzureBuilder, gcp::GoogleCloudStorageBuilder, http::HttpBuilder, local::LocalFileSystem, ObjectStore};
use std::sync::Arc;
use url::Url;
use anyhow::{Result, Context};

/// Factory to create an ObjectStore based on the URI scheme.
/// 
/// Supported schemes:
/// - s3:// -> AmazonS3
/// - az:// or abfs:// -> MicrosoftAzure
/// - gs:// or gcs:// -> GoogleCloudStorage
/// - http:// or https:// -> HttpStore
/// - file:// or /path/to/dir -> LocalFileSystem
pub fn create_object_store(uri: &str) -> Result<Arc<dyn ObjectStore>> {
    let output_store: Arc<dyn ObjectStore>;

    if uri.starts_with('/') || uri.starts_with("file://") {
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        if !std::path::Path::new(path).exists() {
            std::fs::create_dir_all(path).context("Failed to create local directory")?;
        }
        output_store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
    } else {
        let url = Url::parse(uri).context("Invalid URI")?;
        match url.scheme() {
            "s3" => {
                let bucket = url.host_str().context("Missing bucket in S3 URI")?;
                
                let mut builder = AmazonS3Builder::from_env()
                    .with_bucket_name(bucket);

                // Support for custom endpoints (MinIO)
                if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
                    builder = builder
                        .with_endpoint(endpoint)
                        .with_allow_http(true)
                        .with_virtual_hosted_style_request(false);
                }

                let s3 = builder.build().context("Failed to build S3 store")?;
                output_store = Arc::new(s3);
            },
            "az" | "abfs" => {
                let container = url.host_str().context("Missing container in Azure URI")?;
                // Uses AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_ACCESS_KEY from env
                let azure = MicrosoftAzureBuilder::from_env()
                    .with_container_name(container)
                    .build()
                    .context("Failed to build Azure store")?;
                output_store = Arc::new(azure);
            },
            "gs" | "gcs" => {
                let bucket = url.host_str().context("Missing bucket in GCS URI")?;
                // Uses GOOGLE_SERVICE_ACCOUNT or generic token from env
                let gcs = GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket)
                    .build()
                    .context("Failed to build GCS store")?;
                output_store = Arc::new(gcs);
            },
            "http" | "https" => {
                let http = HttpBuilder::new()
                    .with_url(uri)
                    .build()
                    .context("Failed to build HTTP store")?;
                output_store = Arc::new(http);
            },
            _ => anyhow::bail!("Unsupported scheme: {}", url.scheme()),
        }
    }

    Ok(output_store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_local_filesystem_absolute_path() -> Result<()> {
        let temp_dir = tempdir()?;
        let path = temp_dir.path().to_str().unwrap();
        
        let _store = create_object_store(path)?;
        // Verify directory was created
        assert!(std::path::Path::new(path).exists());
        
        Ok(())
    }

    #[test]
    fn test_local_filesystem_file_uri() -> Result<()> {
        let temp_dir = tempdir()?;
        let path = temp_dir.path().to_str().unwrap();
        let uri = format!("file://{}", path);
        
        let _store = create_object_store(&uri)?;
        // Verify directory was created
        assert!(std::path::Path::new(path).exists());
        
        Ok(())
    }

    #[test]
    fn test_local_filesystem_creates_directory() -> Result<()> {
        let temp_dir = tempdir()?;
        let new_dir = temp_dir.path().join("new_subdir");
        let path = new_dir.to_str().unwrap();
        
        // Directory doesn't exist yet
        assert!(!new_dir.exists());
        
        // create_object_store should create it
        let _store = create_object_store(path)?;
        assert!(new_dir.exists());
        
        Ok(())
    }

    #[test]
    fn test_invalid_uri() {
        let result = create_object_store("not a valid uri://");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_scheme() {
        let result = create_object_store("ftp://example.com/bucket");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported scheme"));
    }

    #[test]
    fn test_s3_uri_parsing() {
        // This will fail without AWS credentials, but we can test URI parsing
        let result = create_object_store("s3://my-bucket/prefix");
        // Should fail due to missing credentials, not URI parsing
        // We just verify it doesn't panic and error message is reasonable
        if let Err(e) = result {
            let err_msg = e.to_string();
            // Should not be "Unsupported scheme" error
            assert!(!err_msg.contains("Unsupported scheme"), 
                "S3 URI should be recognized as valid scheme, got: {}", err_msg);
        }
        // If it succeeds (credentials available), that's OK too
    }

    #[test]
    fn test_azure_uri_parsing() {
        // Test both az:// and abfs:// schemes
        let result1 = create_object_store("az://my-container/prefix");
        assert!(result1.is_err());
        
        let result2 = create_object_store("abfs://my-container/prefix");
        assert!(result2.is_err());
        
        // Both should fail due to missing credentials, not URI parsing
        assert!(!result1.unwrap_err().to_string().contains("Unsupported scheme"));
        assert!(!result2.unwrap_err().to_string().contains("Unsupported scheme"));
    }

    #[test]
    fn test_gcs_uri_parsing() {
        // Test both gs:// and gcs:// schemes
        let result1 = create_object_store("gs://my-bucket/prefix");
        let result2 = create_object_store("gcs://my-bucket/prefix");
        
        // Should fail due to missing credentials, not URI parsing
        // We just verify they don't panic and error messages are reasonable
        if let Err(e) = result1 {
            let err_msg = e.to_string();
            assert!(!err_msg.contains("Unsupported scheme"),
                "GS URI should be recognized as valid scheme, got: {}", err_msg);
        }
        if let Err(e) = result2 {
            let err_msg = e.to_string();
            assert!(!err_msg.contains("Unsupported scheme"),
                "GCS URI should be recognized as valid scheme, got: {}", err_msg);
        }
        // If they succeed (credentials available), that's OK too
    }

    #[test]
    fn test_http_uri_parsing() {
        // HTTP store should be created (though it may not be functional without a real server)
        let result = create_object_store("https://example.com/data");
        // This might succeed or fail depending on implementation
        // We just verify it doesn't error on unsupported scheme
        if let Err(e) = result {
            assert!(!e.to_string().contains("Unsupported scheme"));
        }
    }

    #[test]
    fn test_path_normalization() -> Result<()> {
        let temp_dir = tempdir()?;
        let path1 = temp_dir.path().to_str().unwrap();
        let path2 = format!("{}/", path1); // With trailing slash
        
        let _store1 = create_object_store(path1)?;
        let _store2 = create_object_store(&path2)?;
        
        // Both should succeed and create the directory
        assert!(std::path::Path::new(path1).exists());
        
        Ok(())
    }
}
