fn main() {
    println!("cargo:rerun-if-changed=src/core/index/cuda/");
    
    #[cfg(feature = "cuda")]
    {
        use std::process::Command;
        use std::env;

        let out_dir = env::var("OUT_DIR").unwrap();
        
        // Check if nvcc is available
        let status = Command::new("nvcc").arg("--version").output();
        if status.is_err() {
            println!("cargo:warning=nvcc not found, skipping CUDA kernel compilation");
            return;
        }

        // List of CUDA kernels to compile
        let kernels = vec![
            "l2_distance",
            "cosine_distance",
            "inner_product",
            "l1_distance",
            "hamming_distance",
            "jaccard_distance",
        ];

        // Compile each kernel
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
    }
}
