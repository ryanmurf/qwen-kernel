use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../stub/qk_stub.c");
    println!("cargo:rerun-if-changed=../include/qk.h");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap_or_default());
    for (name, define) in [("QK_STUB_LIB", false), ("QK_STUB_LAST_LIB", true)] {
        let lib_path = out_dir.join(if define {
            "libqk_stub_last.so"
        } else {
            "libqk_stub.so"
        });
        let mut cc = Command::new("cc");
        cc.args(["-shared", "-fPIC", "-O2"]);
        if define {
            cc.arg("-DQK_STUB_LAST_OUTPUT");
        }
        let status = cc
            .arg("../stub/qk_stub.c")
            .arg("-o")
            .arg(&lib_path)
            .status()
            .unwrap_or_else(|err| panic!("failed to invoke cc for qk stub: {err}"));
        assert!(status.success(), "failed to compile qk stub");
        println!("cargo:rustc-env={name}={}", lib_path.display());
    }
}
