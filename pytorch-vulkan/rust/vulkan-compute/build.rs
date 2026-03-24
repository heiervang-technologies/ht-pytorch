use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate C header from our public API.
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config.clone())
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(out_dir.join("vulkan_compute.h"));

    // Also write to a known location for the C++ shim to find.
    let header_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("csrc")
        .join("generated");
    std::fs::create_dir_all(&header_dir).ok();
    cbindgen::Builder::new()
        .with_crate(env::var("CARGO_MANIFEST_DIR").unwrap())
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(header_dir.join("vulkan_compute.h"));
}
