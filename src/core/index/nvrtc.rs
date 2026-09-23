// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Version-agnostic `libnvrtc` discovery and JIT compilation.
//!
//! cudarc 0.13's nvrtc loader probes a fixed candidate list
//! (`libnvrtc.so`, `libnvrtc64*.so`, `libnvrtc.so.{12,11,10,1}`) that predates
//! CUDA 13, so pip's `nvidia-*-cu13` wheels (`libnvrtc.so.13`) are never found:
//! the probe panics and the GPU silently falls back to CPU. The demo worked
//! around it by symlinking the probed names and re-execing with
//! `LD_LIBRARY_PATH` (dlopen caches the search path at process start, so it
//! cannot be fixed from inside a running interpreter).
//!
//! This module resolves *whatever* `libnvrtc.so*` is actually installed — any
//! CUDA version, any layout — and compiles the embedded `.cu` sources itself,
//! handing the resulting PTX to cudarc's `load_ptx`. No re-exec, no env-var
//! prefix, no hardcoded version list.
//!
//! Search order (the highest version across all directories wins):
//! 1. `HDB_NVRTC_PATH` (explicit file or directory)
//! 2. the Python `site-packages` reported at module init (pip wheels)
//! 3. `CUDA_HOME` / `CUDA_PATH` / `CUDA_ROOT` (`lib64`, `lib`, `targets/.../lib`)
//! 4. `VIRTUAL_ENV` / `CONDA_PREFIX` / `~/.local` / `/usr` / `/usr/local` wheels
//! 5. `LD_LIBRARY_PATH` entries
//! 6. system multiarch (`/usr/lib/x86_64-linux-gnu`, `/usr/lib64`, `/usr/lib`)

use anyhow::{anyhow, bail, Context, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Opaque nvrtc program handle.
type NvrtcProgram = *mut c_void;
/// nvrtc result code (0 = success).
type NvrtcResult = c_int;

type NvrtcCreateProgram = unsafe extern "C" fn(
    prog: *mut NvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> NvrtcResult;
type NvrtcCompileProgram = unsafe extern "C" fn(
    prog: NvrtcProgram,
    num_opts: c_int,
    opts: *const *const c_char,
) -> NvrtcResult;
type NvrtcGetPtxSize = unsafe extern "C" fn(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
type NvrtcGetPtx = unsafe extern "C" fn(prog: NvrtcProgram, ptx: *mut c_char) -> NvrtcResult;
type NvrtcGetLogSize = unsafe extern "C" fn(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
type NvrtcGetLog = unsafe extern "C" fn(prog: NvrtcProgram, log: *mut c_char) -> NvrtcResult;
type NvrtcDestroyProgram = unsafe extern "C" fn(prog: *mut NvrtcProgram) -> NvrtcResult;

/// `site-packages` reported by the Python interpreter at module init.
static PYTHON_SITE_PACKAGES: OnceLock<PathBuf> = OnceLock::new();

/// Record the interpreter's `site-packages` so pip-installed `nvidia-*-cuXX`
/// wheels are discoverable. Called from the `#[pymodule]` init (where the GIL
/// is held); a no-op if already set.
pub fn set_python_site_packages(path: PathBuf) {
    let _ = PYTHON_SITE_PACKAGES.set(path);
}

/// Subdirectories of `dir` whose name starts with `prefix` (empty prefix = all).
fn subdirs_with_prefix(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
        .map(|e| e.path())
        .collect()
}

/// Best (highest-version) nvrtc library directly inside `dir`.
fn best_in_dir(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_nvrtc_lib(&name) {
            continue;
        }
        let rank = version_rank(&name);
        if best.as_ref().map(|(r, _)| rank > *r).unwrap_or(true) {
            best = Some((rank, entry.path()));
        }
    }
    best.map(|(_, p)| p)
}

/// `nvidia/*/lib` directories under a `site-packages` root.
fn nvidia_lib_dirs(site_packages: &Path) -> Vec<PathBuf> {
    subdirs_with_prefix(&site_packages.join("nvidia"), "")
        .into_iter()
        .map(|pkg| pkg.join("lib"))
        .collect()
}

/// `site-packages` containing this shared object, found via `dladdr`.
///
/// Works without the GIL, so it covers the Python case even when the
/// `#[pymodule]` init never ran, and any layout where the extension sits under
/// a `site-packages` directory.
fn self_site_packages() -> Option<PathBuf> {
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }

    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    // SAFETY: `dladdr` only reads the address and fills `info`.
    let ok = unsafe { dladdr(self_site_packages as *const c_void, &mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) }
        .to_string_lossy()
        .into_owned();
    let mut p = PathBuf::from(path);
    while p.pop() {
        if p.file_name().map(|n| n == "site-packages").unwrap_or(false) {
            return Some(p);
        }
    }
    None
}

/// Python install prefixes that may hold `nvidia-*-cuXX` wheels.
fn python_prefixes() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(p) = std::env::var_os(var) {
            out.push(PathBuf::from(p));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push(PathBuf::from(home).join(".local"));
    }
    out.push(PathBuf::from("/usr"));
    out.push(PathBuf::from("/usr/local"));
    out
}

/// Directories that may contain `libnvrtc.so*`, in priority order.
fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    // 1. Explicit override (file or directory).
    if let Some(p) = std::env::var_os("HDB_NVRTC_PATH") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            dirs.push(p);
        } else if let Some(parent) = p.parent() {
            dirs.push(parent.to_path_buf());
        }
    }

    // 2. The interpreter's site-packages (pip wheels): reported at module init,
    //    or located from this shared object's own path (GIL-free).
    if let Some(sp) = PYTHON_SITE_PACKAGES.get() {
        dirs.extend(nvidia_lib_dirs(sp));
    }
    if let Some(sp) = self_site_packages() {
        dirs.extend(nvidia_lib_dirs(&sp));
    }
    // 2b. PYTHONPATH entries (each is typically a site-packages dir).
    if let Some(pp) = std::env::var_os("PYTHONPATH") {
        for entry in std::env::split_paths(&pp) {
            dirs.extend(nvidia_lib_dirs(&entry));
        }
    }

    // 3. CUDA toolkit installs.
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(root) = std::env::var_os(var) {
            let root = PathBuf::from(root);
            dirs.push(root.join("lib64"));
            dirs.push(root.join("lib"));
            dirs.push(root.join("targets/x86_64-linux/lib"));
        }
    }
    dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
    dirs.push(PathBuf::from("/usr/local/cuda/lib"));
    dirs.push(PathBuf::from("/opt/cuda/lib64"));

    // 4. pip / conda wheels under conventional prefixes.
    for prefix in python_prefixes() {
        for py in subdirs_with_prefix(&prefix.join("lib"), "python3") {
            dirs.extend(nvidia_lib_dirs(&py.join("site-packages")));
        }
    }

    // 5. LD_LIBRARY_PATH.
    if let Some(ld) = std::env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(std::env::split_paths(&ld));
    }

    // 6. System multiarch / default.
    dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/lib64"));
    dirs.push(PathBuf::from("/usr/lib"));

    dirs
}

/// Is `name` an nvrtc shared library (not the `-builtins` companion)?
fn is_nvrtc_lib(name: &str) -> bool {
    name.starts_with("libnvrtc") && name.contains(".so") && !name.starts_with("libnvrtc-builtins")
}

/// Rank a library filename by CUDA version so the highest wins.
///
/// Handles both layouts: `libnvrtc.so.13` / `libnvrtc.so.12.2` and
/// `libnvrtc64_120_0.so` / `libnvrtc64_1200.so`. Unversioned names rank 0.
fn version_rank(name: &str) -> u32 {
    if let Some(idx) = name.find(".so.") {
        let mut parts = name[idx + 4..]
            .split('.')
            .filter_map(|p| p.parse::<u32>().ok());
        let major = parts.next().unwrap_or(0);
        let minor = parts.next().unwrap_or(0);
        return major * 10_000 + minor * 100;
    }
    if let Some(idx) = name.find("nvrtc64_") {
        let digits: String = name[idx + 8..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.len() >= 2 {
            // `120` -> 12, `1200` -> 12, `13` -> 13
            return digits[..2].parse::<u32>().unwrap_or(0) * 10_000;
        }
        if let Ok(major) = digits.parse::<u32>() {
            return major * 10_000;
        }
    }
    0
}

/// Resolve the best available `libnvrtc` shared library, version-agnostically.
pub fn resolve_nvrtc() -> Option<PathBuf> {
    let mut best: Option<(u32, PathBuf)> = None;
    for dir in candidate_dirs() {
        if let Some(candidate) = best_in_dir(&dir) {
            let rank = version_rank(&candidate.file_name()?.to_string_lossy());
            if best.as_ref().map(|(r, _)| rank > *r).unwrap_or(true) {
                best = Some((rank, candidate));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Preload nvrtc's `libnvrtc-builtins*` companion with `RTLD_GLOBAL`.
///
/// nvrtc loads its builtins at compile time via its own `dlopen`, looking the
/// library up by soname. When nvrtc comes from a pip wheel the directory is not
/// on the loader search path, so that lookup fails ("failed to open
/// libnvrtc-builtins.alt.so.13.0"). Loading it ourselves with `RTLD_GLOBAL`
/// registers it under its soname, so nvrtc's lookup succeeds.
fn preload_builtins(nvrtc_path: &Path) {
    static PRELOADED: OnceLock<()> = OnceLock::new();
    PRELOADED.get_or_init(|| {
        let Some(dir) = nvrtc_path.parent() else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("libnvrtc-builtins") || !name.contains(".so") {
                continue;
            }
            // SAFETY: loading a shared library from a directory we just
            // resolved; RTLD_GLOBAL makes it visible to nvrtc's own dlopen.
            if let Ok(lib) = unsafe {
                libloading::os::unix::Library::open(
                    Some(entry.path()),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                )
            } {
                // Keep it loaded for the process lifetime.
                std::mem::forget(lib);
            }
        }
    });
}

/// Read the nvrtc program log (best-effort; used for error messages).
unsafe fn read_log(
    prog: NvrtcProgram,
    get_log_size: NvrtcGetLogSize,
    get_log: NvrtcGetLog,
) -> String {
    let mut size: usize = 0;
    if get_log_size(prog, &mut size) != 0 || size == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if get_log(prog, buf.as_mut_ptr() as *mut c_char) != 0 {
        return String::new();
    }
    CStr::from_ptr(buf.as_ptr() as *const c_char)
        .to_string_lossy()
        .into_owned()
}

/// Compile CUDA C source to PTX using the resolved nvrtc.
///
/// The returned PTX string can be handed to cudarc via
/// `cudarc::nvrtc::Ptx::from_src(..)` + `CudaDevice::load_ptx`.
pub fn compile_ptx(src: &str) -> Result<String> {
    let path = resolve_nvrtc().ok_or_else(|| {
        anyhow!(
            "libnvrtc not found. Install a CUDA toolkit or the `nvidia-cuda-nvrtc-cuXX` \
             wheel, or set HDB_NVRTC_PATH to the library."
        )
    })?;
    compile_ptx_with_path(src, &path)
}

/// Compile CUDA C source to PTX using a specific nvrtc library.
pub fn compile_ptx_with_path(src: &str, path: &Path) -> Result<String> {
    // nvrtc needs its builtins companion on the loader path; preload it.
    preload_builtins(path);

    // SAFETY: `path` points at a real shared library; the symbol signatures
    // below match the nvrtc C API.
    let lib = unsafe { libloading::Library::new(path) }
        .with_context(|| format!("dlopen {}", path.display()))?;

    unsafe {
        let create: NvrtcCreateProgram = *lib
            .get(b"nvrtcCreateProgram\0")
            .context("nvrtcCreateProgram symbol")?;
        let compile: NvrtcCompileProgram = *lib
            .get(b"nvrtcCompileProgram\0")
            .context("nvrtcCompileProgram symbol")?;
        let get_ptx_size: NvrtcGetPtxSize = *lib
            .get(b"nvrtcGetPTXSize\0")
            .context("nvrtcGetPTXSize symbol")?;
        let get_ptx: NvrtcGetPtx = *lib.get(b"nvrtcGetPTX\0").context("nvrtcGetPTX symbol")?;
        let get_log_size: NvrtcGetLogSize = *lib
            .get(b"nvrtcGetProgramLogSize\0")
            .context("nvrtcGetProgramLogSize symbol")?;
        let get_log: NvrtcGetLog = *lib
            .get(b"nvrtcGetProgramLog\0")
            .context("nvrtcGetProgramLog symbol")?;
        let destroy: NvrtcDestroyProgram = *lib
            .get(b"nvrtcDestroyProgram\0")
            .context("nvrtcDestroyProgram symbol")?;

        let src_c = CString::new(src).context("CUDA source contains a NUL byte")?;
        let name_c = CString::new("kernel.cu").unwrap();

        let mut prog: NvrtcProgram = std::ptr::null_mut();
        let rc = create(
            &mut prog,
            src_c.as_ptr(),
            name_c.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if rc != 0 {
            bail!("nvrtcCreateProgram failed (code {rc})");
        }

        // No options: nvrtc picks its default architecture, matching cudarc's
        // `compile_ptx` behaviour.
        let rc = compile(prog, 0, std::ptr::null());
        if rc != 0 {
            let log = read_log(prog, get_log_size, get_log);
            destroy(&mut prog);
            bail!("nvrtcCompileProgram failed (code {rc}):\n{log}");
        }

        let mut size: usize = 0;
        let rc = get_ptx_size(prog, &mut size);
        if rc != 0 {
            destroy(&mut prog);
            bail!("nvrtcGetPTXSize failed (code {rc})");
        }
        let mut buf = vec![0u8; size];
        let rc = get_ptx(prog, buf.as_mut_ptr() as *mut c_char);
        destroy(&mut prog);
        if rc != 0 {
            bail!("nvrtcGetPTX failed (code {rc})");
        }

        // nvrtc writes a NUL-terminated string into `buf`.
        Ok(CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned())
    }
}

/// Test/dev helper: locate an nvrtc library inside a repo-local `.venv*`.
#[cfg(test)]
pub(crate) fn dev_repo_venv_nvrtc() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(".venv") {
            continue;
        }
        for py in subdirs_with_prefix(&entry.path().join("lib"), "python3") {
            for dir in nvidia_lib_dirs(&py.join("site-packages")) {
                if let Some(p) = best_in_dir(&dir) {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_rank_orders_cuda_versions() {
        assert!(version_rank("libnvrtc.so.13") > version_rank("libnvrtc.so.12"));
        assert!(version_rank("libnvrtc.so.12") > version_rank("libnvrtc.so.11"));
        assert!(version_rank("libnvrtc.so.12") > version_rank("libnvrtc.so"));
        assert_eq!(version_rank("libnvrtc.so.13"), 130_000);
        assert_eq!(version_rank("libnvrtc64_120_0.so"), 120_000);
        assert_eq!(version_rank("libnvrtc64_1200.so"), 120_000);
        assert_eq!(version_rank("libnvrtc64_13.so"), 130_000);
    }

    #[test]
    fn is_nvrtc_lib_excludes_builtins() {
        assert!(is_nvrtc_lib("libnvrtc.so.13"));
        assert!(is_nvrtc_lib("libnvrtc64_120_0.so"));
        assert!(!is_nvrtc_lib("libnvrtc-builtins.so.13.0"));
        assert!(!is_nvrtc_lib("libcudart.so.13"));
    }

    #[test]
    fn compiles_a_trivial_kernel() {
        // Prefer the environment, then the repo venv (local dev). Skip cleanly
        // when no nvrtc is available (e.g. CI without CUDA).
        let Some(path) = resolve_nvrtc().or_else(dev_repo_venv_nvrtc) else {
            return;
        };
        let ptx = compile_ptx_with_path("extern \"C\" __global__ void noop_kernel() { }", &path)
            .expect("nvrtc should compile a trivial kernel");
        assert!(ptx.contains(".version"), "expected PTX, got: {ptx:.80}");
    }
}
