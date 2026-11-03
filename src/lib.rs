use cfg_if::cfg_if;
use ort::ep::{self, ExecutionProviderDispatch};

/// Pick the best Execution Provider
pub fn preferred_execution_provider() -> ExecutionProviderDispatch {
    cfg_if! {
        if #[cfg(target_os = "windows")] {
            ep::DirectML::default().build()
        } else if #[cfg(all(target_os = "linux", target_arch = "x86_64"))] {
            ep::WebGPU::default().build()
        } else if #[cfg(all(target_os = "linux", target_arch = "aarch64"))] {
            ep::CUDA::default().build()
        } else if #[cfg(all(target_os = "macos", target_arch = "aarch64"))] {
            ep::WebGPU::default().build()
        } else if #[cfg(all(target_os = "ios", target_arch = "aarch64"))] {
            ep::CoreML::default().build()
        } else if #[cfg(all(target_os = "android", any(target_arch = "x86_64", target_arch = "aarch64")))] {
            ep::NNAPI::default().build()
        } else if #[cfg(all(target_family = "wasm", target_os = "unknown"))] {
            ep::WebGPU::default().build()
        } else {
            compile_error!("GPU sideloading is unsupported on this platform")
        }
    }
}

pub use categories::CATEGORIES;
mod categories;

pub mod timer;
