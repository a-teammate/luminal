// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after heavy search heap churn (hundreds of
// thousands of compiles). jemalloc built with unprefixed symbols
// interposes malloc for the whole process, including dlopened CUDA
// libraries like libnvrtc — a Rust-only global allocator would not.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features `cuda` and `metal` are mutually exclusive");

#[cfg(all(feature = "cuda", feature = "metal"))]
fn main() {}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
#[cfg(all(feature = "metal", not(feature = "cuda"), target_vendor = "apple"))]
use luminal_metal::MetalRuntime;
#[cfg(feature = "reference")]
use qwen::{ReferenceQwen, QwenRunConfig, run_qwen};
#[cfg(any(
    all(feature = "cuda", not(feature = "metal")),
    all(feature = "metal", not(feature = "cuda"), target_vendor = "apple")
))]
use qwen::{QwenRunConfig, Runtime, run_qwen};


// LUMINAL_QWEN_LAYERS caps the layer stack (subset of LAYERS) so the f32
// reference fits in RAM alongside the desktop (nohang kills at ~1.8GB free).
#[cfg(any(feature = "reference", feature = "mojo"))]
fn bench_config() -> qwen::QwenRunConfig {
    let env_parse = |key: &str| {
        std::env::var(key).ok().and_then(|s| s.parse().ok())
    };
    qwen::QwenRunConfig {
        layers: env_parse("LUMINAL_QWEN_LAYERS").unwrap_or(qwen::model::LAYERS),
        max_seq_len: env_parse("LUMINAL_QWEN_MAXSEQ").unwrap_or(2048),
        search_graphs: env_parse("LUMINAL_QWEN_SEARCH_GRAPHS").unwrap_or(500),
        gen_tokens: env_parse("LUMINAL_QWEN_GEN").unwrap_or(16),
        ..Default::default()
    }
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    run_qwen(CudaRuntime::initialize(stream), QwenRunConfig::default()).unwrap();
}

#[cfg(all(feature = "metal", not(feature = "cuda"), target_vendor = "apple"))]
fn main() {
    run_qwen(MetalRuntime::initialize(()), QwenRunConfig::default()).unwrap();
}

#[cfg(all(
    feature = "reference",
    not(any(feature = "cuda", feature = "metal"))
))]
fn main() {
    run_qwen(ReferenceQwen::new(), bench_config()).unwrap();
}

#[cfg(all(
    feature = "mojo",
    not(any(feature = "cuda", feature = "metal", feature = "reference"))
))]
use qwen::{QwenRunConfig, Runtime, run_qwen};

#[cfg(all(
    feature = "mojo",
    not(any(feature = "cuda", feature = "metal", feature = "reference"))
))]
fn main() {
    run_qwen(luminal_mojo::MojoRuntime::new(), bench_config()).unwrap();
}

#[cfg(all(feature = "metal", not(feature = "cuda"), not(target_vendor = "apple")))]
fn main() {
    eprintln!("qwen --features metal requires an Apple target with Metal support.");
}

#[cfg(not(any(
    feature = "cuda",
    all(feature = "metal", target_vendor = "apple"),
    feature = "reference",
    feature = "mojo"
)))]
fn main() {
    eprintln!("select a backend with `--features cuda`, `metal`, or `reference`.");
    std::process::exit(2);
}
