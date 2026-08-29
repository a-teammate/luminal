//! Luminal backend that lowers graphs to Mojo shared libraries via FFI.
//!
//! The runtime generates Mojo source from the LLIR graph, compiles it to a
//! `.so` using `mojo build --emit shared-lib`, then executes by calling the
//! exported C ABI functions through `libloading`.

mod codegen;

use std::ffi::CString;
use std::os::raw::c_void;
use std::process::Command;
use std::time::Duration;

use codegen::{generate_mojo, ExecStep, StepKind};
use luminal::shape::Term;
use libloading::{Library, Symbol};
use luminal::op::Runtime;
use luminal::prelude::*;
use luminal::graph::{BucketLLIR, DimBucket};
use luminal::hlir::Input;

const SAFETY_PAD: usize = 4096;

/// One compiled Mojo graph for a single DimBucket configuration.
struct MojoBucket {
    lib: Library,
    exec_plan: Vec<ExecStep>,
    buffer_sizes: FxHashMap<NodeIndex, usize>,
    input_map: FxHashMap<NodeIndex, NodeIndex>,
    output_map: FxHashMap<NodeIndex, NodeIndex>,
    input_nodes: FxHashSet<NodeIndex>,
    output_nodes: FxHashSet<NodeIndex>,
    representative_dyn_map: FxHashMap<char, usize>,
}

pub struct MojoRuntime {
    // Per-bucket compiled graphs
    buckets: Vec<MojoBucket>,
    dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    active_bucket: usize,

    /// LLIR buffers for the active bucket
    pub buffers: FxHashMap<NodeIndex, Vec<u8>>,

    /// HLIR-level persistent data (weights, KV cache, token IDs).
    /// Keyed by HLIR NodeIndex so it survives bucket switches.
    hlir_data: FxHashMap<NodeIndex, Vec<u8>>,
    dirty: FxHashSet<NodeIndex>,

    /// Pending inputs set before any load_llir call (HLIR-keyed)
    pending_inputs: FxHashMap<NodeIndex, Vec<u8>>,

    /// Dyn map captured during filter_llir_candidate
    captured_dyn_map: FxHashMap<char, usize>,

    /// Actual data size (in f32 elements) per LLIR node
    data_len: FxHashMap<NodeIndex, usize>,
}

impl Default for MojoRuntime {
    fn default() -> Self {
        Self {
            buckets: Vec::new(),
            dim_buckets: FxHashMap::default(),
            active_bucket: 0,
            buffers: FxHashMap::default(),
            hlir_data: FxHashMap::default(),
            dirty: FxHashSet::default(),
            pending_inputs: FxHashMap::default(),
            captured_dyn_map: FxHashMap::default(),
            data_len: FxHashMap::default(),
        }
    }
}

impl MojoRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Production data methods (all HLIR-keyed, stored in hlir_data) ──────

    /// Set F32 input data. Works before and after compilation.
    pub fn set_data_f32(&mut self, hlir_node: NodeIndex, data: &[f32]) {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        self.hlir_data.insert(hlir_node, bytes);
        self.dirty.insert(hlir_node);
    }

    /// Set i32 input data (token IDs, position IDs). Converts to f32 since
    /// all Mojo kernels operate on f32 and gather/scatter read indices as f32.
    pub fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();
        self.hlir_data.insert(id, bytes);
        self.dirty.insert(id);
    }

    /// Zero-initialise a buffer (KV cache init).
    pub fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        self.hlir_data.insert(id, vec![0u8; num_bytes]);
        self.dirty.insert(id);
    }

    /// Remove and return a buffer (for KV cache promote: output → caller).
    pub fn remove_buffer(&mut self, id: NodeIndex) -> Vec<u8> {
        // Try active bucket's output_map first, then input_map
        let llir_node = self.buckets.get(self.active_bucket)
            .and_then(|b| b.output_map.get(&id).or_else(|| b.input_map.get(&id)))
            .copied();
        if let Some(llir) = llir_node {
            self.data_len.remove(&llir);
            self.buffers.remove(&llir).unwrap_or_default()
        } else {
            self.hlir_data.remove(&id).unwrap_or_default()
        }
    }

    /// Set a raw byte buffer as input data (for KV cache promote: caller → input).
    pub fn set_buffer(&mut self, id: NodeIndex, buffer: Vec<u8>) {
        self.hlir_data.insert(id, buffer);
        self.dirty.insert(id);
    }

    /// Load weights from a safetensors file. Converts Bf16/F16 → F32 since
    /// all Mojo kernels operate on f32.
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        use safetensors::SafeTensors;
        use memmap2::MmapOptions;
        use std::fs::File;

        let f = File::open(file_path)
            .unwrap_or_else(|e| panic!("Failed to open safetensors file {file_path}: {e}"));
        let mmap = unsafe { MmapOptions::new().map(&f) }
            .unwrap_or_else(|e| panic!("Failed to mmap {file_path}: {e}"));
        let st = SafeTensors::deserialize(&mmap)
            .unwrap_or_else(|e| panic!("Failed to deserialize safetensors: {e}"));

        for node in cx.graph.node_indices() {
            if let Some(input) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                if let Ok(tensor) = st.tensor(&input.label) {
                    let bytes: Vec<u8> = match tensor.dtype() {
                        safetensors::Dtype::F32 => tensor.data().to_vec(),
                        safetensors::Dtype::BF16 => {
                            let raw = tensor.data();
                            let n = raw.len() / 2;
                            let dst: Vec<f32> = (0..n)
                                .map(|i| {
                                    let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                                    half::bf16::from_bits(bits).to_f32()
                                })
                                .collect();
                            bytemuck::cast_slice(&dst).to_vec()
                        }
                        safetensors::Dtype::F16 => {
                            let raw = tensor.data();
                            let n = raw.len() / 2;
                            let dst: Vec<f32> = (0..n)
                                .map(|i| {
                                    let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                                    half::f16::from_bits(bits).to_f32()
                                })
                                .collect();
                            bytemuck::cast_slice(&dst).to_vec()
                        }
                        _ => tensor.data().to_vec(),
                    };
                    self.hlir_data.insert(node, bytes);
                    self.dirty.insert(node);
                }
            }
        }
    }

    /// Get output data as f32 for a given HLIR tensor.
    pub fn get_f32(&self, hlir_node: NodeIndex) -> Vec<f32> {
        let bucket = &self.buckets[self.active_bucket];
        let llir_node = bucket.output_map.get(&hlir_node)
            .or_else(|| bucket.input_map.get(&hlir_node))
            .copied()
            .unwrap_or(hlir_node);
        let empty: Vec<u8> = Vec::new();
        let bytes = self.buffers.get(&llir_node).unwrap_or(&empty);
        let all_f32: &[f32] = bytemuck::cast_slice(bytes);
        let len = self.data_len.get(&llir_node).copied().unwrap_or(all_f32.len());
        all_f32[..len.min(all_f32.len())].to_vec()
    }

    /// Hook called before each execute — can be used for bucket pre-selection.
    pub fn prepare_execute(&mut self, _dyn_map: &FxHashMap<char, usize>) {}

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Compile Mojo source to a shared library and load it.
    fn compile_and_load(mojo_source: &str) -> Library {
        let tmp_dir = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let source_path = tmp_dir.join(format!("luminal_mojo_{id}.mojo"));
        let lib_path = tmp_dir.join(format!("luminal_mojo_{id}.so"));

        std::fs::write(&source_path, mojo_source)
            .unwrap_or_else(|e| panic!("Failed to write Mojo source: {e}"));

        let output = Command::new("pixi")
            .arg("run")
            .arg("mojo")
            .arg("build")
            .arg("--emit")
            .arg("shared-lib")
            .arg("-o")
            .arg(&lib_path)
            .arg(&source_path)
            .output()
            .unwrap_or_else(|e| panic!("Failed to invoke mojo build: {e}"));

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!(
                "mojo build failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n--- source ---\n{mojo_source}"
            );
        }

        unsafe {
            Library::new(&lib_path)
                .unwrap_or_else(|e| panic!("Failed to load shared library {lib_path:?}: {e}"))
        }
    }

    /// Call luminal_init to initialize Mojo runtime
    fn call_init(lib: &Library) {
        let init_fn: Symbol<extern "C" fn()> = unsafe {
            lib.get(b"luminal_init\0")
                .expect("luminal_init symbol not found")
        };
        init_fn();
    }

    /// Find the bucket whose range covers the requested dyn_map values.
    /// Picks the smallest qualifying bucket.
    fn find_bucket(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        if self.buckets.len() <= 1 {
            return 0;
        }
        let mut best = 0usize;
        let mut best_rep: Option<usize> = None;
        for (i, bucket) in self.buckets.iter().enumerate() {
            // Check if this bucket covers all requested dims
            let covers = dyn_map.iter().all(|(&dim, &val)| {
                if let Some(buckets) = self.dim_buckets.get(&dim) {
                    // Find the bucket index for this dimension from representative_dyn_map
                    let rep_val = bucket.representative_dyn_map.get(&dim).copied().unwrap_or(val);
                    buckets.iter().any(|b| val >= b.min && val <= b.max && rep_val >= b.min && rep_val <= b.max)
                } else {
                    true // dim not bucketed, always covers
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

    /// Switch to a different bucket: save output buffers to hlir_data,
    /// allocate fresh buffers for the new bucket, load hlir_data.
    fn switch_bucket(&mut self, new_bucket: usize) {
        // Save current output buffers back to hlir_data
        let old = &self.buckets[self.active_bucket];
        for (&hlir, &llir) in &old.output_map {
            if let Some(buf) = self.buffers.get(&llir) {
                let len = self.data_len.get(&llir).copied().unwrap_or(buf.len() / 4);
                let byte_len = (len * 4).min(buf.len());
                self.hlir_data.insert(hlir, buf[..byte_len].to_vec());
            }
        }

        self.active_bucket = new_bucket;
        self.allocate_bucket_buffers();
    }

    /// Allocate buffers for the active bucket from buffer_sizes + hlir_data.
    fn allocate_bucket_buffers(&mut self) {
        self.buffers.clear();
        self.data_len.clear();

        let bucket = &self.buckets[self.active_bucket];

        // Intermediate buffers
        for (&node, &size) in &bucket.buffer_sizes {
            let elem_count = size / 4;
            self.data_len.insert(node, elem_count);
            self.buffers.insert(node, vec![0u8; size + SAFETY_PAD]);
        }

        // Input buffers from hlir_data (or pending_inputs)
        for (&hlir_node, &llir_node) in &bucket.input_map {
            if let Some(data) = self.hlir_data.get(&hlir_node).cloned()
                .or_else(|| self.pending_inputs.get(&hlir_node).cloned())
            {
                let elem_count = data.len() / 4;
                self.data_len.insert(llir_node, elem_count);
                let mut buf = data;
                buf.resize(buf.len() + SAFETY_PAD, 0);
                self.buffers.insert(llir_node, buf);
            }
        }

        // Allocate input nodes not covered by hlir_data
        for &node in &bucket.input_nodes {
            self.buffers.entry(node).or_insert_with(|| {
                let size = bucket.buffer_sizes.get(&node).copied().unwrap_or(4);
                self.data_len.entry(node).or_insert(size / 4);
                vec![0u8; size + SAFETY_PAD]
            });
        }
    }

    /// Copy dirty hlir_data into active bucket's LLIR input buffers.
    fn sync_dirty(&mut self) {
        if self.dirty.is_empty() || self.buckets.is_empty() {
            // No buckets loaded yet — data stays in hlir_data/pending_inputs
            // Move hlir_data to pending_inputs for the next load_llir call
            let dirty = std::mem::take(&mut self.dirty);
            for hlir_node in dirty {
                if let Some(data) = self.hlir_data.get(&hlir_node).cloned() {
                    self.pending_inputs.insert(hlir_node, data);
                }
            }
            return;
        }

        let bucket = &self.buckets[self.active_bucket];
        let dirty = std::mem::take(&mut self.dirty);
        for hlir_node in dirty {
            if let Some(data) = self.hlir_data.get(&hlir_node) {
                if let Some(&llir_node) = bucket.input_map.get(&hlir_node) {
                    let mut buf = data.clone();
                    buf.resize(buf.len() + SAFETY_PAD, 0);
                    let elem_count = data.len() / 4;
                    self.data_len.insert(llir_node, elem_count);
                    self.buffers.insert(llir_node, buf);
                }
            }
        }
    }
}

// ── Bucket dispatch helper: execute one step ─────────────────────────────

fn execute_step(
    step: &ExecStep,
    lib: &Library,
    buffers: &mut FxHashMap<NodeIndex, Vec<u8>>,
    data_len: &mut FxHashMap<NodeIndex, usize>,
) {
    match &step.kind {
        StepKind::Binary { a, b, out, .. } => {
            let func: Symbol<extern "C" fn(*const c_void, *const c_void, *mut c_void)> =
                unsafe {
                    let cname = CString::new(step.func_name.as_str()).unwrap();
                    lib.get(cname.as_bytes_with_nul())
                        .unwrap_or_else(|e| panic!("Symbol {} not found: {e}", step.func_name))
                };
            let ptr_a = buffers[a].as_ptr() as *const c_void;
            let ptr_b = buffers[b].as_ptr() as *const c_void;
            let ptr_out = buffers.get_mut(out).unwrap().as_mut_ptr() as *mut c_void;
            func(ptr_a, ptr_b, ptr_out);
        }
        StepKind::Unary { a, out, .. } => {
            let func: Symbol<extern "C" fn(*const c_void, *mut c_void)> = unsafe {
                let cname = CString::new(step.func_name.as_str()).unwrap();
                lib.get(cname.as_bytes_with_nul())
                    .unwrap_or_else(|e| panic!("Symbol {} not found: {e}", step.func_name))
            };
            let ptr_a = buffers[a].as_ptr() as *const c_void;
            let ptr_out = buffers.get_mut(out).unwrap().as_mut_ptr() as *mut c_void;
            func(ptr_a, ptr_out);
        }
        StepKind::Reduce { a, out, .. } => {
            let func: Symbol<extern "C" fn(*const c_void, *mut c_void)> = unsafe {
                let cname = CString::new(step.func_name.as_str()).unwrap();
                lib.get(cname.as_bytes_with_nul())
                    .unwrap_or_else(|e| panic!("Symbol {} not found: {e}", step.func_name))
            };
            let ptr_a = buffers[a].as_ptr() as *const c_void;
            let ptr_out = buffers.get_mut(out).unwrap().as_mut_ptr() as *mut c_void;
            func(ptr_a, ptr_out);
        }
        StepKind::Copy { src, dst } => {
            let src_buf = buffers[src].clone();
            let len = data_len.get(src).copied().unwrap_or(src_buf.len() / 4);
            data_len.insert(*dst, len);
            let copy_bytes = (len * 4).min(src_buf.len());
            let dst_buf = buffers.get_mut(dst).unwrap();
            let n = copy_bytes.min(dst_buf.len());
            dst_buf[..n].copy_from_slice(&src_buf[..n]);
        }
        StepKind::ConstantF32 { out, value } => {
            let buf = buffers.get_mut(out).unwrap();
            let f32_buf = bytemuck::cast_slice_mut::<u8, f32>(buf);
            for v in f32_buf.iter_mut() {
                *v = *value;
            }
        }
        StepKind::RustIota { out, expr, length } => {
            let buf = buffers.get_mut(out).unwrap();
            let f32_buf = bytemuck::cast_slice_mut::<u8, f32>(buf);
            for i in 0..*length {
                f32_buf[i] = eval_iota_expr(expr, i as i64) as f32;
            }
        }
        StepKind::RustGather { indexes, data, out, index_len, phys_map, .. } => {
            let idx_bytes = buffers[indexes].clone();
            let idx_f32: &[f32] = bytemuck::cast_slice(&idx_bytes);
            let data_bytes = buffers[data].clone();
            let data_f32: &[f32] = bytemuck::cast_slice(&data_bytes);
            let out_buf = buffers.get_mut(out).unwrap();
            let out_f32: &mut [f32] = bytemuck::cast_slice_mut(out_buf);
            for i in 0..*index_len {
                let logical_idx = idx_f32[i] as usize;
                let phys_idx = phys_map.get(logical_idx).copied().unwrap_or(logical_idx);
                out_f32[i] = data_f32[phys_idx];
            }
        }
        StepKind::RustScatter { out, dest, indexes, src, dest_len, index_len, dest_phys, idx_phys, src_phys } => {
            let dest_bytes = buffers[dest].clone();
            let mut out_f32: Vec<f32> = bytemuck::cast_slice(&dest_bytes).to_vec();
            let idx_bytes = buffers[indexes].clone();
            let idx_f32: &[f32] = bytemuck::cast_slice(&idx_bytes);
            let src_bytes = buffers[src].clone();
            let src_f32: &[f32] = bytemuck::cast_slice(&src_bytes);
            for i in 0..*index_len {
                let idx_read = idx_phys.get(i).copied().unwrap_or(i);
                let logical_idx = idx_f32[idx_read] as usize;
                let dest_write = dest_phys.get(logical_idx).copied().unwrap_or(logical_idx);
                let src_read = src_phys.get(i).copied().unwrap_or(i);
                if dest_write < *dest_len {
                    out_f32[dest_write] = src_f32[src_read];
                }
            }
            let out_buf = buffers.get_mut(out).unwrap();
            let out_slice: &mut [f32] = bytemuck::cast_slice_mut(out_buf);
            let n = out_slice.len().min(out_f32.len());
            out_slice[..n].copy_from_slice(&out_f32[..n]);
        }
    }
}

impl Runtime for MojoRuntime {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(_: Self::CompileArg) -> Self {
        Self::default()
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        let dyn_map = if self.captured_dyn_map.is_empty() {
            FxHashMap::default()
        } else {
            self.captured_dyn_map.clone()
        };
        let result = generate_mojo(llir_graph, &dyn_map);

        let lib = Self::compile_and_load(&result.mojo_source);
        Self::call_init(&lib);

        let bucket = MojoBucket {
            lib,
            exec_plan: result.exec_plan,
            buffer_sizes: result.buffer_sizes.clone(),
            input_map: result.input_hlir_to_llir,
            output_map: result.output_hlir_to_llir,
            input_nodes: result.input_nodes,
            output_nodes: result.output_nodes,
            representative_dyn_map: dyn_map,
        };

        self.buckets = vec![bucket];
        self.active_bucket = 0;
        self.allocate_bucket_buffers();

        // Flush remaining pending_inputs
        let pending = std::mem::take(&mut self.pending_inputs);
        for (hlir_node, data) in pending {
            if let Some(&llir_node) = self.buckets[0].input_map.get(&hlir_node) {
                let mut buf = data;
                self.data_len.insert(llir_node, buf.len() / 4);
                buf.resize(buf.len() + SAFETY_PAD, 0);
                self.buffers.insert(llir_node, buf);
            }
        }
    }

    fn load_llir_buckets(
        &mut self,
        dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        if bucket_llirs.len() == 1 {
            self.load_llir(&bucket_llirs[0].2);
            return;
        }

        self.dim_buckets = dim_buckets.clone();
        self.buckets.clear();

        for (_bucket_indices, representative_dyn_map, llir_graph) in bucket_llirs {
            let result = generate_mojo(llir_graph, representative_dyn_map);
            let lib = Self::compile_and_load(&result.mojo_source);
            Self::call_init(&lib);

            self.buckets.push(MojoBucket {
                lib,
                exec_plan: result.exec_plan,
                buffer_sizes: result.buffer_sizes.clone(),
                input_map: result.input_hlir_to_llir,
                output_map: result.output_hlir_to_llir,
                input_nodes: result.input_nodes,
                output_nodes: result.output_nodes,
                representative_dyn_map: representative_dyn_map.clone(),
            });
        }
        self.active_bucket = 0;
        self.allocate_bucket_buffers();
    }

    fn filter_llir_candidate(
        &mut self,
        _llir_graph: &LLIRGraph,
        context: luminal::op::CandidateFilterContext<'_>,
    ) -> luminal::op::CandidateFilterResult {
        self.captured_dyn_map = context.dyn_map.clone();
        luminal::op::CandidateFilterResult::accept()
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        // Select bucket
        let new_active = self.find_bucket(dyn_map);
        if new_active != self.active_bucket && !self.buckets.is_empty() {
            self.switch_bucket(new_active);
        }

        // Sync dirty hlir_data → LLIR buffers
        self.sync_dirty();

        // Execute all steps from active bucket
        let active = self.active_bucket;
        let bucket = &self.buckets[active];
        let lib = &bucket.lib;
        let exec_plan = &bucket.exec_plan;
        let output_nodes = bucket.output_nodes.clone();

        // We need to borrow buffers/data_len mutably but bucket immutably,
        // so clone the plan references and use the free function.
        let plan: Vec<ExecStep> = exec_plan.clone();
        for step in &plan {
            execute_step(step, lib, &mut self.buffers, &mut self.data_len);
        }

        // Retain only output buffers
        self.buffers.retain(|k, _| output_nodes.contains(k));
    }

    fn profile(
        &mut self,
        _llir_graph: &LLIRGraph,
        _dyn_map: &FxHashMap<char, usize>,
        _trials: usize,
        _timeout: Option<Duration>,
        _early_stop: Option<(Self::ProfileMetric, f64)>,
    ) -> (Self::ProfileMetric, String) {
        (Duration::ZERO, "0 ms".to_string())
    }
}

/// Evaluate an Iota expression with z = i, returning an integer value.
fn eval_iota_expr(terms: &[Term], i: i64) -> i64 {
    let mut stack: Vec<i64> = Vec::new();
    for term in terms {
        match term {
            Term::Num(n) => stack.push(*n as i64),
            Term::Var(_) => stack.push(i),
            Term::Add => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a + b);
            }
            Term::Sub => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a - b);
            }
            Term::Mul => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a * b);
            }
            Term::Div => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a / b);
            }
            Term::CeilDiv => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push((a + b - 1) / b);
            }
            Term::Mod => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a % b);
            }
            Term::Max => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a.max(b));
            }
            Term::Min => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(a.min(b));
            }
            Term::And => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(((a != 0) && (b != 0)) as i64);
            }
            Term::Or => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(((a != 0) || (b != 0)) as i64);
            }
            Term::Gte => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push((a >= b) as i64);
            }
            Term::Lt => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push((a < b) as i64);
            }
        }
    }
    stack.pop().unwrap_or(0)
}
