// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::{Context, Result};
use object_store::{
    aws::AmazonS3Builder, azure::MicrosoftAzureBuilder, gcp::GoogleCloudStorageBuilder,
    http::HttpBuilder, local::LocalFileSystem, memory::InMemory, ObjectStore,
};
use std::collections::HashMap;
use std::sync::Arc;
use url::Url;

/// Process-wide registry of in-memory stores, keyed by URI.
///
/// `memory://<name>` resolves to a single shared `InMemory` store per name, so
/// multiple `Table` handles opened with the same URI share one store — the
/// shared-object-store model used by the WS3 concurrency harness. Without the
/// registry each call would create an isolated store and the handles would not
/// see each other's commits.
static MEMORY_STORES: once_cell::sync::Lazy<
    parking_lot::Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
> = once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(HashMap::new()));

/// Whether the SSRF guard is active. Off by default so local catalogs
/// (`http://localhost:8181`) keep working; enable in production with
/// `BSDB_SSRF_GUARD=1`.
pub fn ssrf_guard_enabled() -> bool {
    matches!(
        std::env::var("BSDB_SSRF_GUARD").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Reject URIs whose host is a private, loopback, or link-local address.
///
/// Basic SSRF (Server-Side Request Forgery) guard applied at the trust
/// boundary where user-supplied URIs enter the engine (`register_external`,
/// catalog creation). DNS rebinding is not covered here.
pub fn validate_external_uri(uri: &str) -> Result<()> {
    let url = Url::parse(uri).context("Invalid URI")?;
    let host = match url.host() {
        Some(h) => h,
        None => return Ok(()), // no host → nothing to check (e.g. `memory://`)
    };
    match host {
        url::Host::Domain(d) => {
            if d == "localhost" {
                anyhow::bail!("SSRF guard: host 'localhost' is not allowed in external URIs");
            }
        }
        url::Host::Ipv4(ip) => {
            if ip.is_loopback() || ip.is_private() || ip.is_unspecified() || ip.is_link_local() {
                anyhow::bail!(
                    "SSRF guard: private/loopback/link-local IPv4 address {ip} is not allowed in external URIs"
                );
            }
        }
        url::Host::Ipv6(ip) => {
            if ip.is_loopback() || ip.is_unspecified() {
                anyhow::bail!(
                    "SSRF guard: loopback/unspecified IPv6 address {ip} is not allowed in external URIs"
                );
            }
        }
    }
    Ok(())
}

/// Factory to create an ObjectStore based on the URI scheme.
///
/// Supported schemes:
/// - s3:// -> AmazonS3
/// - az:// or abfs:// -> MicrosoftAzure
/// - gs:// or gcs:// -> GoogleCloudStorage
/// - http:// or https:// -> HttpStore
/// - file:// or /path/to/dir -> LocalFileSystem
pub fn create_object_store(uri: &str) -> Result<Arc<dyn ObjectStore>> {
    // SSRF guard: reject non-file URIs that point at internal hosts.
    // `memory://` and `file://` are always local; remote schemes must
    // pass the address check below.
    if ssrf_guard_enabled()
        && uri.contains("://")
        && !uri.starts_with("file://")
        && !uri.starts_with("memory://")
    {
        validate_external_uri(uri)?;
    }

    let output_store: Arc<dyn ObjectStore>;

    if uri.starts_with('/') || uri.starts_with("file://") || !uri.contains("://") {
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        if !std::path::Path::new(path).exists() {
            std::fs::create_dir_all(path).context("Failed to create local directory")?;
        }
        output_store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
    } else {
        let url = Url::parse(uri).context("Invalid URI")?;
        match url.scheme() {
            "s3" | "s3a" => {
                let bucket = url.host_str().context("Missing bucket in S3 URI")?;

                let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);

                // Support for custom endpoints (RustFS)
                if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
                    builder = builder
                        .with_endpoint(endpoint)
                        .with_allow_http(true)
                        .with_virtual_hosted_style_request(false);
                }

                let s3 = builder.build().context("Failed to build S3 store")?;
                let path = url.path().trim_start_matches('/');
                if !path.is_empty() {
                    output_store =
                        Arc::new(object_store::prefix::PrefixStore::new(s3, path.to_string()));
                } else {
                    output_store = Arc::new(s3);
                }
            }
            "az" | "abfs" => {
                let container = url.host_str().context("Missing container in Azure URI")?;
                // Uses AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_ACCESS_KEY from env
                let azure = MicrosoftAzureBuilder::from_env()
                    .with_container_name(container)
                    .build()
                    .context("Failed to build Azure store")?;
                let path = url.path().trim_start_matches('/');
                if !path.is_empty() {
                    output_store = Arc::new(object_store::prefix::PrefixStore::new(
                        azure,
                        path.to_string(),
                    ));
                } else {
                    output_store = Arc::new(azure);
                }
            }
            "gs" | "gcs" => {
                let bucket = url.host_str().context("Missing bucket in GCS URI")?;
                // Uses GOOGLE_SERVICE_ACCOUNT or generic token from env
                let gcs = GoogleCloudStorageBuilder::from_env()
                    .with_bucket_name(bucket)
                    .build()
                    .context("Failed to build GCS store")?;
                let path = url.path().trim_start_matches('/');
                if !path.is_empty() {
                    output_store = Arc::new(object_store::prefix::PrefixStore::new(
                        gcs,
                        path.to_string(),
                    ));
                } else {
                    output_store = Arc::new(gcs);
                }
            }
            "http" | "https" => {
                let http = HttpBuilder::new()
                    .with_url(uri)
                    .build()
                    .context("Failed to build HTTP store")?;
                output_store = Arc::new(http);
            }
            "memory" => {
                // Shared in-memory store keyed by the full URI (see MEMORY_STORES).
                let mut map = MEMORY_STORES.lock();
                output_store = map
                    .entry(uri.to_string())
                    .or_insert_with(|| Arc::new(InMemory::new()))
                    .clone();
            }
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
    fn ssrf_guard_rejects_internal_hosts() {
        // Loopback / private / link-local must be rejected.
        assert!(validate_external_uri("http://localhost:8181").is_err());
        assert!(validate_external_uri("http://127.0.0.1:9000").is_err());
        assert!(validate_external_uri("http://10.0.0.5/").is_err());
        assert!(validate_external_uri("http://192.168.1.1/").is_err());
        assert!(validate_external_uri("http://172.16.0.1/").is_err());
        assert!(validate_external_uri("http://169.254.169.254/").is_err());
        assert!(validate_external_uri("http://[::1]:8080/").is_err());
        // Public hosts and host-less schemes are allowed.
        assert!(validate_external_uri("https://catalog.example.com").is_ok());
        assert!(validate_external_uri("s3://my-bucket/prefix").is_ok());
        assert!(validate_external_uri("memory://shared").is_ok());
    }

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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported scheme"));
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
            assert!(
                !err_msg.contains("Unsupported scheme"),
                "S3 URI should be recognized as valid scheme, got: {}",
                err_msg
            );
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
        assert!(!result1
            .unwrap_err()
            .to_string()
            .contains("Unsupported scheme"));
        assert!(!result2
            .unwrap_err()
            .to_string()
            .contains("Unsupported scheme"));
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
            assert!(
                !err_msg.contains("Unsupported scheme"),
                "GS URI should be recognized as valid scheme, got: {}",
                err_msg
            );
        }
        if let Err(e) = result2 {
            let err_msg = e.to_string();
            assert!(
                !err_msg.contains("Unsupported scheme"),
                "GCS URI should be recognized as valid scheme, got: {}",
                err_msg
            );
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

    #[tokio::test]
    async fn test_object_store_quotes() -> Result<()> {
        use crate::core::manifest::types::{PartitionField, PartitionSpec};
        use std::collections::HashMap;

        let temp_dir = tempfile::tempdir()?;
        let path = temp_dir.path().to_str().unwrap();
        let store = create_object_store(path)?;

        let spec = PartitionSpec {
            spec_id: 0,
            fields: vec![PartitionField {
                source_ids: vec![1],
                source_id: Some(1),
                field_id: None,
                name: "category".to_string(),
                transform: "identity".to_string(),
            }],
        };

        // Test normal string
        let mut values1 = HashMap::new();
        values1.insert("category".to_string(), serde_json::json!("A"));
        let path1 = spec.partition_to_path(&values1);
        assert_eq!(path1, "category=A");

        // Test string with space and special characters
        let mut values2 = HashMap::new();
        values2.insert("category".to_string(), serde_json::json!("A B/C"));
        let path2 = spec.partition_to_path(&values2);
        assert_eq!(path2, "category=A%20B%2FC");

        // Create the directory on disk using the generated path
        let dir_on_disk = temp_dir.path().join(&path2);
        std::fs::create_dir_all(&dir_on_disk)?;
        let file_path = dir_on_disk.join("test.parquet");
        std::fs::File::create(&file_path)?;

        // Try to access it via object store
        let relative_path_str = format!("{}/test.parquet", path2);
        let pq_path = object_store::path::Path::parse(&relative_path_str).unwrap();

        let res = store.head(&pq_path).await;
        println!("store.head result for encoded path: {:?}", res);
        assert!(res.is_ok());

        Ok(())
    }
}
