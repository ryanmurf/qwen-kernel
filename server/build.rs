use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../stub/qk_stub.c");
    println!("cargo:rerun-if-changed=../include/qk.h");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap_or_default());
    let lib_path = out_dir.join("libqk_stub.so");
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-O2", "../stub/qk_stub.c", "-o"])
        .arg(&lib_path)
        .status()
        .unwrap_or_else(|err| panic!("failed to invoke cc for qk stub: {err}"));
    if !status.success() {
        panic!("failed to compile qk stub");
    }
    println!("cargo:rustc-env=QK_STUB_LIB={}", lib_path.display());
}
