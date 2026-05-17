fn main() {
    // Specify exactly which files/directories to watch for changes
    // This prevents Cargo from scanning the entire directory tree
    println!("cargo:rerun-if-changed=ui/main.slint");
    println!("cargo:rerun-if-changed=ui/");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=assets/");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");

    slint_build::compile("ui/main.slint").unwrap();
}
