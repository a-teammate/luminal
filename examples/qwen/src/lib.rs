pub mod hf;
pub mod model;

use hf::prepare_hf_model;
pub use luminal::prelude::Runtime;
use luminal::prelude::*;
use luminal_tracing::luminal_filter;
use model::*;
use rustc_hash::FxHashSet;
use std::{error::Error, io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const EOS_TOKEN: u32 = 151645; // <|im_end|>
const STOP_TOKEN: u32 = 151643; // <|endoftext|>

pub struct QwenRunConfig {
    pub repo_id: String,
    pub max_seq_len: usize,
    pub gen_tokens: usize,
    pub search_graphs: usize,
    pub prompt: String,
    pub repetition_penalty: f32,
    pub layers: usize,
}

fn qwen3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

impl Default for QwenRunConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen3-0.6B".to_string(),
            max_seq_len: 2048,
            gen_tokens: 16,
            search_graphs: 500,
            prompt: "Explain what a neural network is in a paragraph.".to_string(),
            repetition_penalty: 1.05,
            layers: LAYERS,
        }
    }
}

pub trait QwenRuntime: Runtime<ExecReturn = ()> {
    type Buffer;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str);
    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>);
    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize);
    // TODO(aliasing): the CUDA backend now supports user-owned aliased
    // state buffers (CudaRuntime::alias_state), which replaces the per-step
    // remove_buffer/set_buffer promote below. This example keeps the legacy
    // promote because it also targets the Metal backend, which has no
    // aliasing API yet.
    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer;
    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer);
    fn get_f32(&self, id: NodeIndex) -> Vec<f32>;

    fn prepare_execute(&mut self, _dyn_map: &DynMap) {}
}

#[cfg(feature = "cuda")]
impl QwenRuntime for luminal_cuda_lite::runtime::CudaRuntime {
    type Buffer = luminal_cuda_lite::cudarc::driver::CudaSlice<u8>;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        luminal_cuda_lite::runtime::CudaRuntime::load_safetensors(self, cx, file_path);
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        luminal_cuda_lite::runtime::CudaRuntime::set_data(self, id, data);
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        luminal_cuda_lite::runtime::CudaRuntime::set_zeros(self, id, num_bytes);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        luminal_cuda_lite::runtime::CudaRuntime::remove_buffer(self, id)
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        luminal_cuda_lite::runtime::CudaRuntime::set_buffer(self, id, buffer);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        luminal_cuda_lite::runtime::CudaRuntime::get_f32(self, id)
    }
}

#[cfg(feature = "metal")]
impl QwenRuntime for luminal_metal::MetalRuntime {
    type Buffer = luminal_metal::Buffer;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        luminal_metal::MetalRuntime::load_safetensors(self, cx, file_path);
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        luminal_metal::MetalRuntime::set_data(self, id, data);
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        luminal_metal::MetalRuntime::set_zeros(self, id, num_bytes);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        luminal_metal::MetalRuntime::remove_buffer(self, id)
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        luminal_metal::MetalRuntime::set_buffer(self, id, buffer);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        luminal_metal::MetalRuntime::get_f32(self, id)
    }
}

#[cfg(feature = "mojo")]
impl QwenRuntime for luminal_mojo::MojoRuntime {
    type Buffer = Vec<u8>;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        luminal_mojo::MojoRuntime::load_safetensors(self, cx, file_path);
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        luminal_mojo::MojoRuntime::set_i32_data(self, id, data);
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        luminal_mojo::MojoRuntime::set_zeros(self, id, num_bytes);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        luminal_mojo::MojoRuntime::remove_buffer(self, id)
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        luminal_mojo::MojoRuntime::set_buffer(self, id, buffer);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        luminal_mojo::MojoRuntime::get_f32(self, id)
    }

    fn prepare_execute(&mut self, dyn_map: &luminal::shape::DynMap) {
        luminal_mojo::MojoRuntime::prepare_execute(self, dyn_map);
    }
}

/// CPU reference baseline. `ReferenceRuntime` compiles one LLIR program and
/// has no per-execute bucket selection, and bucketed compiles const-fold
/// singleton dims into each bucket's program — so a bucketed graph needs one
/// `ReferenceRuntime` per bucket, switched by the execute-time dyn map.
/// Tensors (weights, KV state, inputs) live in a shared store and are synced
/// lazily into each bucket runtime on first use.
#[cfg(feature = "reference")]
use luminal::hlir::{Input, Output};
#[cfg(feature = "reference")]
use luminal::search::extract_one;

#[cfg(feature = "reference")]
struct ReferenceBucket {
    representative_dyn_map: DynMap,
    rt: luminal::hlir::ReferenceRuntime,
    /// HLIR nodes already copied from the shared store into this runtime.
    synced: FxHashSet<NodeIndex>,
    /// HLIR ids that exist as Input nodes in this bucket's program (a bucket
    /// may fold away tensors other buckets need).
    input_nodes: FxHashSet<usize>,
}

#[cfg(feature = "reference")]
pub struct ReferenceQwen {
    dim_buckets: FxHashMap<luminal::shape::Symbol, Vec<DimBucket>>,
    buckets: Vec<ReferenceBucket>,
    active: usize,
    /// Cross-bucket tensor store: (HLIR node, latest data).
    shared: Vec<(NodeIndex, luminal::hlir::ReferenceData)>,
    shared_nodes: FxHashSet<NodeIndex>,
}

#[cfg(feature = "reference")]
impl ReferenceQwen {
    pub fn new() -> Self {
        Self {
            dim_buckets: Default::default(),
            buckets: vec![],
            active: 0,
            shared: vec![],
            shared_nodes: Default::default(),
        }
    }

    /// Keep the latest data per HLIR node, preserving insertion order.
    fn shared_upsert(&mut self, id: NodeIndex, data: luminal::hlir::ReferenceData) {
        if self.shared_nodes.insert(id) {
            self.shared.push((id, data));
        } else if let Some(slot) = self.shared.iter_mut().find(|(n, _)| *n == id) {
            slot.1 = data;
        }
    }

    /// Copy shared tensors a bucket runtime hasn't seen yet into it,
    /// skipping nodes this bucket's program doesn't have.
    fn sync_bucket(&mut self, idx: usize) {
        let pending: Vec<(NodeIndex, luminal::hlir::ReferenceData)> = self
            .shared
            .iter()
            .filter(|(n, _)| {
                !self.buckets[idx].synced.contains(n)
                    && self.buckets[idx].input_nodes.contains(&n.index())
            })
            .map(|(n, d)| (*n, d.clone()))
            .collect();
        for (node, data) in pending {
            self.buckets[idx].rt.set_data(node, data);
            self.buckets[idx].synced.insert(node);
        }
    }

    fn find_bucket(&self, dyn_map: &DynMap) -> usize {
        if self.buckets.len() <= 1 {
            return 0;
        }
        let mut best = 0usize;
        let mut best_rep: Option<usize> = None;
        for (i, bucket) in self.buckets.iter().enumerate() {
            let covers = dyn_map.iter().all(|(&dim, &val)| {
                if let Some(buckets) = self.dim_buckets.get(&dim) {
                    let rep_val =
                        bucket.representative_dyn_map.get(&dim).copied().unwrap_or(val);
                    buckets.iter().any(|b| {
                        val >= b.min && val <= b.max && rep_val >= b.min && rep_val <= b.max
                    })
                } else {
                    true
                }
            });
            if covers {
                let rep_sum: usize = bucket.representative_dyn_map.values().sum();
                if best_rep.is_none() || rep_sum < best_rep.unwrap() {
                    best = i;
                    best_rep = Some(rep_sum);
                }
            }
        }
        best
    }

    fn set_f32_impl(&mut self, id: NodeIndex, data: Vec<f32>) {
        let has_buckets = !self.buckets.is_empty()
            && self.buckets[self.active].input_nodes.contains(&id.index());
        let active = self.active;
        self.shared_upsert(id, luminal::hlir::ReferenceData::F32(data.clone()));
        if has_buckets {
            self.buckets[active].rt.set_data(id, data);
            self.buckets[active].synced.insert(id);
        }
    }

    fn active_rt(&self) -> &luminal::hlir::ReferenceRuntime {
        &self.buckets[self.active].rt
    }

    fn active_rt_mut(&mut self) -> &mut luminal::hlir::ReferenceRuntime {
        &mut self.buckets[self.active].rt
    }

    /// Find the reference-graph node for a persisted tensor's read side
    /// (consumers of a persist tensor read the twin created for it).
    fn persist_read_node(rt: &luminal::hlir::ReferenceRuntime, id: NodeIndex) -> NodeIndex {
        rt.graph
            .node_indices()
            .find(|n| {
                if let Some(output) = (**rt.graph[*n]).as_any().downcast_ref::<Output>() {
                    output.node == id.index()
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("{id:?} has no read twin in the reference graph"))
    }

    fn reference_to_bytes(data: luminal::hlir::ReferenceData) -> Vec<u8> {
        match data {
            luminal::hlir::ReferenceData::F32(v) => bytemuck::cast_slice(&v).to_vec(),
            luminal::hlir::ReferenceData::Int(v) => bytemuck::cast_slice(&v).to_vec(),
            luminal::hlir::ReferenceData::U8(v) => v,
            other => panic!("unsupported reference data {other:?}"),
        }
    }
}

#[cfg(feature = "reference")]
impl Default for ReferenceQwen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "reference")]
impl Runtime for ReferenceQwen {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();

    fn initialize(_: Self::CompileArg) -> Self {
        Self::new()
    }

    fn compile(
        &mut self,
        space: &luminal::search::SearchSpace,
        dyn_map: &DynMap,
        options: &CompileOptions,
        rng: &mut dyn rand::RngCore,
    ) {
        self.dim_buckets = options.dim_buckets.clone();
        self.buckets = space
            .bucket_contexts(dyn_map)
            .into_iter()
            .map(|ctx| {
                let llir = extract_one(space, &ctx, rng);
                let mut rt = luminal::hlir::ReferenceRuntime::default();
                rt.load_llir(&llir);
                let input_nodes: FxHashSet<usize> = rt
                    .graph
                    .node_indices()
                    .filter_map(|n| {
                        (**rt.graph[n]).as_any().downcast_ref::<Input>().map(|i| i.node)
                    })
                    .collect();
                if std::env::var("LUMINAL_QWEN_DEBUG").as_deref() == Ok("1") {
                    let mut ids: Vec<_> = input_nodes.iter().copied().collect();
                    ids.sort_unstable();
                    eprintln!(
                        "[reference bucket {}] reps {:?}, {} input nodes: {:?}",
                        ctx.index,
                        ctx.representative_dyn_map,
                        input_nodes.len(),
                        ids,
                    );
                }
                ReferenceBucket {
                    representative_dyn_map: ctx.representative_dyn_map.clone(),
                    rt,
                    synced: Default::default(),
                    input_nodes,
                }
            })
            .collect();
        self.active = 0;
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        if self.buckets.is_empty() {
            self.buckets.push(ReferenceBucket {
                representative_dyn_map: Default::default(),
                rt: Default::default(),
                synced: Default::default(),
                input_nodes: Default::default(),
            });
        }
        self.active = 0;
        self.buckets[0].rt.load_llir(llir_graph);
    }

    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn {
        let idx = self.find_bucket(dyn_map);
        self.sync_bucket(idx);
        self.active = idx;
        self.buckets[idx].rt.execute(dyn_map)
    }
}

#[cfg(feature = "reference")]
impl QwenRuntime for ReferenceQwen {
    type Buffer = Vec<u8>;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        use safetensors::SafeTensors;
        let f = std::fs::File::open(file_path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(input) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                if let Ok(tensor) = st.tensor(&input.label) {
                    let f32_vec: Vec<f32> = match tensor.dtype() {
                        safetensors::Dtype::F32 => bytemuck::cast_slice(tensor.data()).to_vec(),
                        safetensors::Dtype::BF16 => bytemuck::cast_slice::<u8, u16>(tensor.data())
                            .iter()
                            .map(|&bits| half::bf16::from_bits(bits).to_f32())
                            .collect(),
                        other => panic!("unsupported tensor dtype {other:?}"),
                    };
                    self.shared_upsert(node, luminal::hlir::ReferenceData::F32(f32_vec));
                }
            }
        }
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        let has_buckets = !self.buckets.is_empty()
            && self.buckets[self.active].input_nodes.contains(&id.index());
        let active = self.active;
        self.shared_upsert(id, luminal::hlir::ReferenceData::Int(data.clone()));
        if has_buckets {
            self.buckets[active].rt.set_data(id, data);
            self.buckets[active].synced.insert(id);
        }
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        self.set_f32_impl(id, vec![0f32; num_bytes / 4]);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        let local = Self::persist_read_node(self.active_rt(), id);
        let data = self.active_rt_mut().buffers.remove(&local).unwrap();
        let bytes = Self::reference_to_bytes(data.clone());
        self.shared_upsert(id, data);
        bytes
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        let data = bytemuck::cast_slice::<u8, f32>(&buffer).to_vec();
        self.set_f32_impl(id, data);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        self.active_rt().get_f32(id).clone()
    }

    fn prepare_execute(&mut self, _dyn_map: &DynMap) {}
}

pub fn run_qwen<R>(mut runtime: R, config: QwenRunConfig) -> Result<(), Box<dyn Error>>
where
    R: QwenRuntime + 'static,
{
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .try_init();

    let model_dir = prepare_hf_model(&config.repo_id)?;
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| err as Box<dyn Error>)?;
    let prompt = qwen3_chat_prompt(&config.prompt);
    let prompt_tokens = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|err| err as Box<dyn Error>)?
        .get_ids()
        .to_vec();

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, config.max_seq_len, config.layers);
    let (logits, cache_outputs) =
        Qwen::init(&mut cx, config.layers).forward(input, token_ids, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }
    let prompt_len = prompt_tokens.len();
    // Bucketing specializes one compiled program per (s, p) range. Kernels
    // bake the representative dims, so every bucket is a point bucket whose
    // representative equals the runtime value (see the kv_cache_lifecycle
    // tests). The decode positions are known: prompt_len + step. The
    // reference runtime runs a single dynamic program, so it needs the
    // unbucketed (symbolic) compile; LUMINAL_QWEN_BUCKETS=0 opts out.
    let use_buckets = std::env::var("LUMINAL_QWEN_BUCKETS")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut compile_options = if use_buckets {
        let mut opts = CompileOptions::default();
        let mut s_buckets = vec![DimBucket::new(1, 1)];
        if prompt_len > 1 {
            s_buckets.push(DimBucket::new(prompt_len, prompt_len));
        }
        opts = opts.dim_buckets('s', &s_buckets);
        opts
    } else {
        CompileOptions::default()
    };
    let decode_steps = config.gen_tokens;
    if use_buckets {
        let mut p_buckets = vec![DimBucket::new(0, 0)];
        for step in 0..decode_steps {
            let p = prompt_len + step;
            if p <= config.max_seq_len - 1 {
                // Buckets must be unique; skip positions colliding with 0.
                if p > 0 {
                    p_buckets.push(DimBucket::new(p, p));
                }
            }
        }
        compile_options = compile_options.dim_buckets('p', &p_buckets);
    }

    println!("Loading weights...");
    let weights_path = model_dir.join("model_combined_bf16_v1.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // KV cache element size follows the pipeline dtype.
    let cache_bytes = N_KV_HEADS * config.max_seq_len * HEAD_DIM * MODEL_DTYPE_SIZE;
    for i in 0..config.layers {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    println!("Compiling...");
    cx.set_dim('s', prompt_len);
    cx.set_dim('p', 0);
    eprintln!("[{} prefill start]", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
    runtime.set_i32_data(input.id, vec![1; prompt_len]);
    runtime.set_i32_data(token_ids.id, (0..prompt_len as i32).collect::<Vec<_>>());
    compile_options = compile_options.search_graph_limit(config.search_graphs);
    runtime = cx.compile(runtime, compile_options);
    cx.drop_search_space();

    if std::env::var("LUMINAL_QWEN_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "[hlir] input={:?} token_ids={:?} first k_cache={:?} first v_cache={:?}",
            input.id,
            token_ids.id,
            kv_cache.k_caches.first().map(|t| t.id),
            kv_cache.v_caches.first().map(|t| t.id),
        );
    }

    for i in 0..config.layers {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    let prompt_len = prompt_tokens.len();
    let mut prev_seq = 0usize;
    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, config.gen_tokens
    );

    let mut generated = 0usize;
    let mut sentence = Vec::new();

    if config.gen_tokens > 0 && prompt_len > 0 {
        let start = std::time::Instant::now();

        cx.set_dim('s', prompt_len);
        cx.set_dim('p', 0);

        runtime.set_i32_data(
            input.id,
            prompt_tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        );
        runtime.set_i32_data(token_ids.id, (0..prompt_len as i32).collect::<Vec<_>>());
        runtime.prepare_execute(&cx.dyn_map);

        runtime.execute(&cx.dyn_map);
        if std::env::var("LUMINAL_QWEN_DEBUG").as_deref() == Ok("1") {
            eprintln!("[{} prefill execute done]", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        }
        let logits_data = runtime.get_f32(logits.id);

        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(k_out.id);
            let v_buf = runtime.remove_buffer(v_out.id);
            runtime.set_buffer(kv_cache.k_caches[layer_idx].id, k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx].id, v_buf);
        }

        prev_seq = prompt_len;
        fwd_durations.push(start.elapsed());

        // The head runs on the gathered last row only: logits are [1, V].
        let row_start = 0;
        let mut last_row = logits_data[row_start..row_start + VOCAB_SIZE].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= config.repetition_penalty;
            } else {
                *logit *= config.repetition_penalty;
            }
        }
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        sentence = vec![next_token];
        seen_tokens.insert(next_token);
        generated = 1;

        if next_token != EOS_TOKEN && next_token != STOP_TOKEN {
            let decoded = tokenizer
                .decode(&[next_token], true)
                .map_err(|err| err as Box<dyn Error>)?;
            print!("{}", decoded);
            std::io::stdout().flush()?;
        }
    }

    while generated < config.gen_tokens && !sentence.is_empty() {
        let start = std::time::Instant::now();
        let seq_len = sentence.len();
        let current_token = sentence[0];

        if current_token == EOS_TOKEN || current_token == STOP_TOKEN {
            break;
        }

        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        runtime.set_i32_data(
            input.id,
            sentence.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        );
        runtime.set_i32_data(
            token_ids.id,
            (prev_seq as i32..(seq_len + prev_seq) as i32).collect::<Vec<_>>(),
        );
        runtime.prepare_execute(&cx.dyn_map);

        if std::env::var("LUMINAL_QWEN_DEBUG").as_deref() == Ok("1") {
            eprintln!("[{} decode {generated}] executing s={seq_len} p={prev_seq}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        }
        runtime.execute(&cx.dyn_map);
        if std::env::var("LUMINAL_QWEN_DEBUG").as_deref() == Ok("1") {
            eprintln!("[{} decode {generated}] execute done", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        }
        let logits_data = runtime.get_f32(logits.id);

        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(k_out.id);
            let v_buf = runtime.remove_buffer(v_out.id);
            runtime.set_buffer(kv_cache.k_caches[layer_idx].id, k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx].id, v_buf);
        }

        prev_seq += seq_len;
        fwd_durations.push(start.elapsed());

        let mut last_row = logits_data[logits_data.len() - VOCAB_SIZE..].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= config.repetition_penalty;
            } else {
                *logit *= config.repetition_penalty;
            }
        }
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        sentence = vec![next_token];
        seen_tokens.insert(next_token);
        generated += 1;

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        let decoded = tokenizer
            .decode(&[next_token], true)
            .map_err(|err| err as Box<dyn Error>)?;
        print!("{}", decoded);
        std::io::stdout().flush()?;
    }
    println!();

    let decode_durations: Vec<_> = fwd_durations.iter().skip(1).collect();
    if decode_durations.len() > 2 {
        println!(
            "  TTFT: {:.2} ms",
            fwd_durations[..1].iter().sum::<Duration>().as_secs_f64() * 1e3
        );
        println!(
            "  TPOT: {:.2} ms",
            (decode_durations.iter().skip(1).copied().sum::<Duration>()
                / (decode_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1_000.
        );
    }

    Ok(())
}
