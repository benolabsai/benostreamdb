use std::process::Command;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/core/index/cuda/");
    println!("cargo:rerun-if-changed=.git/HEAD");
    
    // Get git tag version for version injection
    let version = get_git_version();
    println!("cargo:rustc-env=HYPERSTREAMDB_VERSION={}", version);
    
    // Write version constant to version.rs for use in code
    let out_dir = env::var("OUT_DIR").unwrap();
    let version_file = Path::new(&out_dir).join("version.rs");
    fs::write(&version_file, format!(r#"pub const VERSION: &str = "{}";"#, version))
        .expect("Failed to write version.rs");
    
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    if target_os == "macos" {
        println!("cargo:rustc-cfg=feature=\"mps\"");
    } else {
        // Auto-detect CUDA, ROCm, or fallback to Intel OpenCL
        let has_nvcc = Command::new("nvcc").arg("--version").output().is_ok();
        
        if has_nvcc {
            println!("cargo:rustc-cfg=feature=\"cuda\"");
            
            let out_dir = env::var("OUT_DIR").unwrap();
            let kernels = vec![
                "l2_distance",
                "cosine_distance",
                "inner_product",
                "l1_distance",
                "hamming_distance",
                "jaccard_distance",
                "kmeans_assignment",
            ];

            for kernel_name in kernels {
                let kernel_src = format!("src/core/index/cuda/{}.cu", kernel_name);
                let kernel_ptx = format!("{}/{}.ptx", out_dir, kernel_name);

                let status = Command::new("nvcc")
                    .args(&["-ptx", &kernel_src, "-o", &kernel_ptx])
                    .status()
                    .expect(&format!("Failed to execute nvcc for {}", kernel_name));

                if !status.success() {
                    panic!("nvcc failed for {}", kernel_name);
                }
            }
        } else if Command::new("rocminfo").output().is_ok() {
            println!("cargo:rustc-cfg=feature=\"rocm\"");
        } else {
            println!("cargo:rustc-cfg=feature=\"intel\"");
        }
    }
}

fn get_git_version() -> String {
    // Try to get the latest git tag
    if let Ok(output) = Command::new("git")
        .args(&["describe", "--tags", "--abbrev=0"])
        .output()
    {
        if output.status.success() {
            let tag = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_string();
            // Remove 'v' prefix if present (e.g., v0.1.2 -> 0.1.2)
            if tag.starts_with('v') {
                return tag[1..].to_string();
            }
            return tag;
        }
    }
    
    // Fall back to getting current commit hash
    if let Ok(output) = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let hash = String::from_utf8_lossy(&output.stdout)
                .trim()
                .to_string();
            return format!("0.1.0-{}", hash);
        }
    }
    
    // Final fallback to Cargo.toml version
    "0.1.0".to_string()
}
