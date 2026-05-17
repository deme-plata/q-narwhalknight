fn main() {
    // Only build the OpenCL shim on Linux when the gpu-mining feature is enabled.
    // The shim replaces the hard libOpenCL.so dependency: it dlopens the real
    // library at runtime if present (GPU mode), otherwise returns error codes
    // so the miner falls back to CPU automatically.
    let is_linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let gpu_feature = std::env::var("CARGO_FEATURE_GPU_MINING").is_ok();

    if is_linux && gpu_feature {
        let shim_src = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("opencl_shim")
            .join("opencl_shim.c");

        if shim_src.exists() {
            // Compiles opencl_shim.c → libOpenCL.a in OUT_DIR and emits
            // cargo:rustc-link-search=native=$OUT_DIR automatically.
            cc::Build::new()
                .file(&shim_src)
                .opt_level(2)
                .compile("OpenCL");

            // Force static resolution of all OpenCL symbols from our shim.
            // This prevents libOpenCL.so from ending up in DT_NEEDED.
            println!("cargo:rustc-link-lib=static=OpenCL");

            // --as-needed: if the dynamic libOpenCL.so resolves no new symbols
            // (they're all already resolved by our static shim) it's omitted from
            // DT_NEEDED, making the binary start fine without libOpenCL installed.
            println!("cargo:rustc-link-arg=-Wl,--as-needed");

            // dlopen + pthread_once used by the shim
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
    }
}
