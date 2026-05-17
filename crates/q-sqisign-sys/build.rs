use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let dst = cmake::Config::new("sqisign-ref")
        .define("SQISIGN_BUILD_TYPE", "ref")
        .define("GMP_LIBRARY", "SYSTEM")
        .define("ENABLE_TESTS", "OFF")
        .define("ENABLE_SIGN", "ON")
        // -fno-lto overrides cmake's CheckIPOSupported which would otherwise
        // enable GCC LTO. GCC-LTO bitcode objects are incompatible with
        // rust-lld (LLVM linker). -fPIC is needed for position-independent code.
        .define("CMAKE_C_FLAGS", "-fPIC -fno-lto")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF")
        .build_target("sqisign_lvl1_nistapi")
        .build();

    let build_dir = dst.join("build");
    let out_dir = env::var("OUT_DIR").unwrap();
    let combined_lib = Path::new(&out_dir).join("libsqisign_combined.a");

    // Collect all component static libraries
    let lib_paths: Vec<_> = [
        "src/libsqisign_lvl1_nistapi.a",
        "src/libsqisign_lvl1.a",
        "src/signature/ref/lvl1/libsqisign_signature_lvl1.a",
        "src/id2iso/ref/lvl1/libsqisign_id2iso_lvl1.a",
        "src/verification/ref/lvl1/libsqisign_verification_lvl1.a",
        "src/quaternion/ref/generic/libsqisign_quaternion_generic.a",
        "src/hd/ref/lvl1/libsqisign_hd_lvl1.a",
        "src/ec/ref/lvl1/libsqisign_ec_lvl1.a",
        "src/precomp/ref/lvl1/libsqisign_precomp_lvl1.a",
        "src/gf/ref/lvl1/libsqisign_gf_lvl1.a",
        "src/mp/ref/generic/libsqisign_mp_generic.a",
        "src/common/generic/libsqisign_common_sys.a",
    ]
    .iter()
    .map(|p| build_dir.join(p))
    .collect();

    // Verify all libraries exist
    for path in &lib_paths {
        assert!(
            path.exists(),
            "Missing library: {}",
            path.display()
        );
    }

    // Merge all static libraries into one combined archive using a MRI script.
    // This avoids circular dependency issues with the linker.
    let mri_script_path = Path::new(&out_dir).join("merge.mri");
    let mut mri_script = String::from("CREATE ");
    mri_script.push_str(combined_lib.to_str().unwrap());
    mri_script.push('\n');
    for path in &lib_paths {
        mri_script.push_str("ADDLIB ");
        mri_script.push_str(path.to_str().unwrap());
        mri_script.push('\n');
    }
    mri_script.push_str("SAVE\nEND\n");
    std::fs::write(&mri_script_path, &mri_script).expect("failed to write MRI script");

    let status = Command::new("ar")
        .arg("-M")
        .stdin(std::fs::File::open(&mri_script_path).unwrap())
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar -M failed to merge static libraries");

    // Link the combined archive + GMP
    println!("cargo:rustc-link-search=native={}", out_dir);
    // Use +whole-archive to force the linker to include all object files
    // from the archive. Without this, lld drops unreferenced objects from .a
    // files before resolving symbols from Rust, causing undefined symbol errors.
    println!("cargo:rustc-link-lib=static:+whole-archive=sqisign_combined");
    println!("cargo:rustc-link-lib=gmp");

    // Re-run if the C source changes
    println!("cargo:rerun-if-changed=sqisign-ref/src");
    println!("cargo:rerun-if-changed=sqisign-ref/include");
    println!("cargo:rerun-if-changed=sqisign-ref/CMakeLists.txt");
}
