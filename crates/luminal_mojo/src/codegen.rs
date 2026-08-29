//! Mojo source code generation from Luminal LLIR graphs.
//!
//! Each LLIR compute node becomes one `@export` Mojo function.
//! The generated .mojo file is compiled to a shared library and loaded via FFI.

use std::fmt::Write;

use itertools::Itertools;

use crate::gemm::{MojoGemmLLIR, MojoOp};

use luminal::hlir::{
    Add, Cast, Constant, Exp2, Gather, Iota, Log2, MaxReduce, Mod, Mul, LessThan,
    Recip, Scatter, Sin, Sqrt, SumReduce,
    Input, Output, ReferenceOp,
};
use luminal::prelude::petgraph::{
    algo::toposort,
    visit::EdgeRef,
    Direction,
};
use luminal::prelude::*;
use luminal::shape::Term;

/// A single step in the execution plan.
#[derive(Clone)]
pub struct ExecStep {
    /// Function symbol name in the shared library (e.g. "op_3")
    pub func_name: String,
    /// What kind of operation this is
    pub kind: StepKind,
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum StepKind {
    /// Binary elementwise: (a_ptr, b_ptr, out_ptr)
    Binary { a: NodeIndex, b: NodeIndex, out: NodeIndex, op: BinaryOp },
    /// Unary elementwise: (a_ptr, out_ptr)
    Unary { a: NodeIndex, out: NodeIndex, op: UnaryOp },
    /// Reduction: (a_ptr, out_ptr)
    Reduce { a: NodeIndex, out: NodeIndex, op: ReduceOp },
    /// Constant scalar: just fill the buffer in Rust, no Mojo call needed
    ConstantF32 { out: NodeIndex, value: f32 },
    /// Copy from source to output (for Output nodes)
    Copy { src: NodeIndex, dst: NodeIndex },
    /// Rust-side Iota: generate sequential f32 values
    RustIota { out: NodeIndex, expr: Vec<Term>, length: usize },
    /// Rust-side Gather: indexes[n] -> data[phys_map[indexes[n]]]
    RustGather { indexes: NodeIndex, data: NodeIndex, out: NodeIndex, index_len: usize, data_len: usize, phys_map: Vec<usize> },
    /// Rust-side Scatter: out = copy of dest, then out[dest_phys[indexes[idx_phys[i]]]] = src[src_phys[i]]
    RustScatter { out: NodeIndex, dest: NodeIndex, indexes: NodeIndex, src: NodeIndex, dest_len: usize, index_len: usize, dest_phys: Vec<usize>, idx_phys: Vec<usize>, src_phys: Vec<usize> },
    /// Fused GEMM: out[b,m,n] = relu(a[b,m,k] @ b[b,k,n] + bias[n]); batch is
    /// baked into the emitted kernel, so the FFI signature stays pointer-only
    Gemm { a: NodeIndex, b: NodeIndex, bias: Option<NodeIndex>, out: NodeIndex },
}

#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Mul,
    Mod,
    LessThan,
}

#[derive(Clone, Copy)]
pub enum UnaryOp {
    Exp2,
    Log2,
    Sin,
    Recip,
    Sqrt,
}

#[derive(Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
}

/// Result of codegen: the Mojo source string and the execution plan.
pub struct CodegenResult {
    pub mojo_source: String,
    pub exec_plan: Vec<ExecStep>,
    /// Buffer sizes in bytes for each node
    pub buffer_sizes: FxHashMap<NodeIndex, usize>,
    /// Which nodes are inputs (HLIR Input nodes)
    pub input_nodes: FxHashSet<NodeIndex>,
    /// Which nodes are outputs (HLIR Output nodes)
    pub output_nodes: FxHashSet<NodeIndex>,
    /// Mapping from HLIR Input node index to LLIR node index (for set_data)
    pub input_hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    /// Mapping from HLIR Output node index to LLIR node index (for get)
    pub output_hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
}

/// Resolve an Expression to a concrete usize using dyn_map.
fn resolve_expr(expr: &Expression, dyn_map: &FxHashMap<char, usize>) -> usize {
    expr.exec(dyn_map)
        .unwrap_or_else(|| panic!("Failed to resolve expression {expr:?} with dyn_map {dyn_map:?}"))
}

/// Resolve a list of Expressions to concrete usizes.
fn resolve_exprs(exprs: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Vec<usize> {
    exprs.iter().map(|e| resolve_expr(e, dyn_map)).collect()
}

/// Convert an Expression to Mojo code, resolving dyn_map vars and replacing
/// remaining variables (like 'z') with the given replacement string.
fn expr_to_mojo(expr: &Expression, dyn_map: &FxHashMap<char, usize>, z_replace: &str) -> String {
    let resolved = expr.resolve_vars(dyn_map);
    let mut stack: Vec<String> = Vec::new();
    for term in resolved.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push(n.to_string()),
            Term::Var(_) => stack.push(z_replace.to_string()),
            // NOTE: Pop order MUST match Expression::exec_stack:
            //   a = pop() (top), b = pop() (second), result = op(a, b)
            Term::Add => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("({a} + {b})"));
            }
            Term::Sub => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("({a} - {b})"));
            }
            Term::Mul => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("({a} * {b})"));
            }
            Term::Div => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("({a} // {b})"));
            }
            Term::Mod => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("({a} % {b})"));
            }
            Term::Max => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("max({a}, {b})"));
            }
            Term::Min => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                stack.push(format!("min({a}, {b})"));
            }
            _ => panic!("Unsupported term in expression: {term:?}"),
        }
    }
    stack.pop().unwrap_or_else(|| "0".to_string())
}

/// Generate Mojo source code from an LLIR graph.
///
/// `dyn_map` resolves dynamic dimension symbols to concrete values.
pub fn generate_mojo(
    llir_graph: &LLIRGraph,
    dyn_map: &FxHashMap<char, usize>,
) -> CodegenResult {
    let topo = toposort(llir_graph, None).unwrap_or_else(|e| panic!("Cycle in LLIR: {:?}", e));

    let mut mojo_funcs = Vec::new();
    let mut exec_plan = Vec::new();
    let mut buffer_sizes: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut input_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut output_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut hlir_to_llir_inputs: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    let mut hlir_to_llir_outputs: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();

    for (step_idx, &node) in topo.iter().enumerate() {
        let llir_op = &llir_graph[node];

        // Check for Input
        if let Some(input) = llir_op.to_op::<Input>() {
            input_nodes.insert(node);
            hlir_to_llir_inputs.insert(NodeIndex::new(input.node), node);
            continue;
        }

        // Check for Output
        if let Some(output) = llir_op.to_op::<Output>() {
            output_nodes.insert(node);
            hlir_to_llir_outputs.insert(NodeIndex::new(output.node), node);

            // Output copies from its single input
            let src = llir_graph
                .edges_directed(node, Direction::Incoming)
                .next()
                .map(|e| e.source())
                .unwrap();
            let size = buffer_sizes.get(&src).copied().unwrap_or(0);
            buffer_sizes.insert(node, size);
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::Copy { src, dst: node },
            });
            continue;
        }

        // Fused GEMM emitted by the backend's egglog rules
        if let Some(mojo_op) = llir_op.to_dialect::<dyn MojoOp>() {
            if let Some(gemm) = mojo_op.as_ref().as_any().downcast_ref::<MojoGemmLLIR>() {
                let m = resolve_expr(&gemm.m, dyn_map);
                let n = resolve_expr(&gemm.n, dyn_map);
                let k = resolve_expr(&gemm.k, dyn_map);
                let batch = resolve_expr(&gemm.batch, dyn_map);
                let inputs: Vec<NodeIndex> = llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect();
                let bias = if gemm.bias { Some(inputs[2]) } else { None };
                buffer_sizes.insert(node, batch * m * n * 4);
                let fn_name = format!("op_{step_idx}");
                mojo_funcs.push(gen_gemm(&fn_name, batch, m, n, k, gemm.bias, gemm.relu));
                exec_plan.push(ExecStep {
                    func_name: fn_name,
                    kind: StepKind::Gemm { a: inputs[0], b: inputs[1], bias, out: node },
                });
                continue;
            }
        }

        // Try to downcast to dyn ReferenceOp
        let op_ref = llir_op.to_dialect::<dyn ReferenceOp>();
        if op_ref.is_none() {
            continue;
        }
        let op_any = op_ref.unwrap().as_ref().as_any();

        // Get input node indices (sorted by edge order)
        let inputs: Vec<NodeIndex> = llir_graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();

        let byte_size = |n: usize| n * 4; // F32 = 4 bytes

        // Dispatch on op type
        if let Some(add) = op_any.downcast_ref::<Add>() {
            let shape = resolve_exprs(&add.shape, dyn_map);
            let a_idx_exprs: Vec<String> = add.a_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let b_idx_exprs: Vec<String> = add.b_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product::<usize>();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_binary_strided(&fn_name, "+", &shape, &a_idx_exprs, &b_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Binary { a: inputs[0], b: inputs[1], out: node, op: BinaryOp::Add },
            });
        } else if let Some(mul) = op_any.downcast_ref::<Mul>() {
            let shape = resolve_exprs(&mul.shape, dyn_map);
            let a_idx_exprs: Vec<String> = mul.a_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let b_idx_exprs: Vec<String> = mul.b_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product::<usize>();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_binary_strided(&fn_name, "*", &shape, &a_idx_exprs, &b_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Binary { a: inputs[0], b: inputs[1], out: node, op: BinaryOp::Mul },
            });
        } else if let Some(op) = op_any.downcast_ref::<Exp2>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_unary_strided(&fn_name, "exp2(a.unsafe_load({a}))", &shape, &a_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Unary { a: inputs[0], out: node, op: UnaryOp::Exp2 },
            });
        } else if let Some(op) = op_any.downcast_ref::<Log2>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_unary_strided(&fn_name, "log2(a.unsafe_load({a}))", &shape, &a_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Unary { a: inputs[0], out: node, op: UnaryOp::Log2 },
            });
        } else if let Some(op) = op_any.downcast_ref::<Sin>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_unary_strided(&fn_name, "sin(a.unsafe_load({a}))", &shape, &a_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Unary { a: inputs[0], out: node, op: UnaryOp::Sin },
            });
        } else if let Some(op) = op_any.downcast_ref::<Recip>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_unary_strided(&fn_name, "1.0 / a.unsafe_load({a})", &shape, &a_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Unary { a: inputs[0], out: node, op: UnaryOp::Recip },
            });
        } else if let Some(op) = op_any.downcast_ref::<Sqrt>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_unary_strided(&fn_name, "sqrt(a.unsafe_load({a}))", &shape, &a_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Unary { a: inputs[0], out: node, op: UnaryOp::Sqrt },
            });
        } else if let Some(op) = op_any.downcast_ref::<SumReduce>() {
            let out_shape = resolve_exprs(&op.shape, dyn_map);
            let stride_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let iters = resolve_expr(&op.iters, dyn_map);
            let iter_stride_expr = expr_to_mojo(&op.iter_stride, dyn_map, "k");
            let n_out: usize = out_shape.iter().product::<usize>().max(1);
            let size = byte_size(n_out);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_reduce(&fn_name, "acc", "0.0", "acc += val", &out_shape, &stride_exprs, iters, &iter_stride_expr));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Reduce { a: inputs[0], out: node, op: ReduceOp::Sum },
            });
        } else if let Some(op) = op_any.downcast_ref::<MaxReduce>() {
            let out_shape = resolve_exprs(&op.shape, dyn_map);
            let stride_exprs: Vec<String> = op.strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let iters = resolve_expr(&op.iters, dyn_map);
            let iter_stride_expr = expr_to_mojo(&op.iter_stride, dyn_map, "k");
            let n_out: usize = out_shape.iter().product::<usize>().max(1);
            let size = byte_size(n_out);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_reduce(&fn_name, "best", "-1e30", "best = max(best, val)", &out_shape, &stride_exprs, iters, &iter_stride_expr));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Reduce { a: inputs[0], out: node, op: ReduceOp::Max },
            });
        } else if let Some(c) = op_any.downcast_ref::<Constant>() {
            let size = byte_size(1);
            buffer_sizes.insert(node, size);
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::ConstantF32 { out: node, value: c.0 },
            });
        } else if let Some(op) = op_any.downcast_ref::<Mod>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.a_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let b_idx_exprs: Vec<String> = op.b_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product::<usize>();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            mojo_funcs.push(gen_binary_strided(&fn_name, "%", &shape, &a_idx_exprs, &b_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Binary { a: inputs[0], b: inputs[1], out: node, op: BinaryOp::Mod },
            });
        } else if let Some(op) = op_any.downcast_ref::<LessThan>() {
            let shape = resolve_exprs(&op.shape, dyn_map);
            let a_idx_exprs: Vec<String> = op.a_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let b_idx_exprs: Vec<String> = op.b_strides.iter()
                .enumerate()
                .map(|(d, e)| expr_to_mojo(e, dyn_map, &format!("i{d}")))
                .collect();
            let n: usize = shape.iter().product::<usize>();
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            let fn_name = format!("op_{step_idx}");
            // LessThan: output 1.0 if a < b, else 0.0
            mojo_funcs.push(gen_binary_cmp(&fn_name, "<", &shape, &a_idx_exprs, &b_idx_exprs));
            exec_plan.push(ExecStep {
                func_name: fn_name,
                kind: StepKind::Binary { a: inputs[0], b: inputs[1], out: node, op: BinaryOp::LessThan },
            });
        } else if let Some(op) = op_any.downcast_ref::<Iota>() {
            // Iota(Expression, Expression) - generates sequential values
            let length = resolve_expr(&op.1, dyn_map);
            let size = byte_size(length);
            buffer_sizes.insert(node, size);
            let resolved = op.0.resolve_vars(dyn_map);
            let terms = resolved.terms.read().clone();
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::RustIota { out: node, expr: terms, length },
            });
        } else if let Some(op) = op_any.downcast_ref::<Cast>() {
            // Cast(Expression, DType) - dtype conversion
            // All data is treated as f32 in Mojo, so Cast is identity copy
            let n = resolve_expr(&op.0, dyn_map);
            let size = byte_size(n);
            buffer_sizes.insert(node, size);
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::Copy { src: inputs[0], dst: node },
            });
        } else if let Some(op) = op_any.downcast_ref::<Gather>() {
            // Gather: inputs[0]=indexes (int), inputs[1]=data
            let index_len: usize = resolve_exprs(&op.index_shape, dyn_map).iter().product();
            let data_shape = resolve_exprs(&op.data_shape, dyn_map);
            let data_len: usize = data_shape.iter().product();
            // Precompute physical index mapping using data strides (as expressions)
            let data_stride_exprs: Vec<Expression> = if op.data_strides.is_empty() {
                contiguous_strides(&data_shape).iter().map(|&s| Expression::from(s as i64)).collect()
            } else {
                op.data_strides.iter().map(|e| e.resolve_vars(dyn_map)).collect()
            };
            let phys_map = build_strided_index_map_expr(&data_shape, &data_stride_exprs);
            let size = byte_size(index_len);
            buffer_sizes.insert(node, size);
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::RustGather { indexes: inputs[0], data: inputs[1], out: node, index_len, data_len, phys_map },
            });
        } else if let Some(op) = op_any.downcast_ref::<Scatter>() {
            // Scatter: inputs[0]=dest, inputs[1]=indexes, inputs[2]=src
            let dest_shape = resolve_exprs(&op.dest_shape, dyn_map);
            let dest_len: usize = dest_shape.iter().product();
            let index_shape = resolve_exprs(&op.index_shape, dyn_map);
            let index_len: usize = index_shape.iter().product();
            let dest_stride_exprs: Vec<Expression> = op.dest_strides.iter()
                .map(|e| e.resolve_vars(dyn_map)).collect();
            let idx_stride_exprs: Vec<Expression> = op.index_strides.iter()
                .map(|e| e.resolve_vars(dyn_map)).collect();
            let src_stride_exprs: Vec<Expression> = op.src_strides.iter()
                .map(|e| e.resolve_vars(dyn_map)).collect();
            let dest_phys = build_strided_index_map_expr(&dest_shape, &dest_stride_exprs);
            let idx_phys = build_strided_index_map_expr(&index_shape, &idx_stride_exprs);
            let src_phys = build_strided_index_map_expr(&index_shape, &src_stride_exprs);
            let size = byte_size(dest_len);
            buffer_sizes.insert(node, size);
            exec_plan.push(ExecStep {
                func_name: String::new(),
                kind: StepKind::RustScatter { out: node, dest: inputs[0], indexes: inputs[1], src: inputs[2], dest_len, index_len, dest_phys, idx_phys, src_phys },
            });
        } else {
            panic!("Unsupported LLIR op type in Mojo codegen: {op_any:?}");
        }
    }

    let mojo_source = assemble_mojo_source(&mojo_funcs);

    CodegenResult {
        mojo_source,
        exec_plan,
        buffer_sizes,
        input_nodes,
        output_nodes,
        input_hlir_to_llir: hlir_to_llir_inputs,
        output_hlir_to_llir: hlir_to_llir_outputs,
    }
}

fn assemble_mojo_source(funcs: &[String]) -> String {
    let mut src = String::new();
    writeln!(src, "from std.runtime import initialize_runtime").unwrap();
    writeln!(src, "from std.memory import OpaquePointer").unwrap();
    writeln!(src, "from std.origin import MutUntrackedOrigin").unwrap();
    writeln!(src, "from std.math import exp2, log2, sin, sqrt, max").unwrap();
    writeln!(src).unwrap();
    writeln!(src, "@export(\"luminal_init\")").unwrap();
    writeln!(src, "def luminal_init() abi(\"C\") -> None:").unwrap();
    writeln!(src, "    initialize_runtime()").unwrap();
    writeln!(src).unwrap();
    for f in funcs {
        writeln!(src, "{f}").unwrap();
    }
    src
}

/// Generate a strided binary op function.
/// `a_idx_exprs` and `b_idx_exprs` are Mojo expressions for the index into each
/// input, parameterized by loop variables i0, i1, ...
fn gen_binary_strided(
    fn_name: &str,
    op: &str,  // "+" or "*"
    shape: &[usize],
    a_idx_exprs: &[String],
    b_idx_exprs: &[String],
) -> String {
    let ndim = shape.len();

    // Build nested loops
    let mut body = String::new();
    let mut indent = "    ".to_string();
    for d in 0..ndim {
        writeln!(body, "{indent}for i{d} in range({}):", shape[d]).unwrap();
        indent.push_str("    ");
    }
    let a_idx = a_idx_exprs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" + ");
    let b_idx = b_idx_exprs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" + ");
    let a_idx = if a_idx.is_empty() { "0".to_string() } else { a_idx };
    let b_idx = if b_idx.is_empty() { "0".to_string() } else { b_idx };
    writeln!(body, "{indent}out.unsafe_store(out_pos, a.unsafe_load({a_idx}) {op} b.unsafe_load({b_idx}))").unwrap();
    writeln!(body, "{indent}out_pos += 1").unwrap();

    format!(
r##"@export("{fn_name}")
def {fn_name}(
    a_ptr: OpaquePointer[MutUntrackedOrigin],
    b_ptr: OpaquePointer[MutUntrackedOrigin],
    out_ptr: OpaquePointer[MutUntrackedOrigin],
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var b = b_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out_pos = 0
{body}"##
    )
}

/// Generate a strided binary comparison function (output 1.0/0.0 as f32).
fn gen_binary_cmp(
    fn_name: &str,
    cmp: &str,
    shape: &[usize],
    a_idx_exprs: &[String],
    b_idx_exprs: &[String],
) -> String {
    let ndim = shape.len();

    let mut body = String::new();
    let mut indent = "    ".to_string();
    for d in 0..ndim {
        writeln!(body, "{indent}for i{d} in range({}):", shape[d]).unwrap();
        indent.push_str("    ");
    }
    let a_idx = a_idx_exprs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" + ");
    let b_idx = b_idx_exprs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" + ");
    let a_idx = if a_idx.is_empty() { "0".to_string() } else { a_idx };
    let b_idx = if b_idx.is_empty() { "0".to_string() } else { b_idx };
    writeln!(body, "{indent}if a.unsafe_load({a_idx}) {cmp} b.unsafe_load({b_idx}):").unwrap();
    writeln!(body, "{indent}    out.unsafe_store(out_pos, 1.0)").unwrap();
    writeln!(body, "{indent}else:").unwrap();
    writeln!(body, "{indent}    out.unsafe_store(out_pos, 0.0)").unwrap();
    writeln!(body, "{indent}out_pos += 1").unwrap();

    format!(
r##"@export("{fn_name}")
def {fn_name}(
    a_ptr: OpaquePointer[MutUntrackedOrigin],
    b_ptr: OpaquePointer[MutUntrackedOrigin],
    out_ptr: OpaquePointer[MutUntrackedOrigin],
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var b = b_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out_pos = 0
{body}"##
    )
}

/// Generate a unary elementwise op function (contiguous).
fn gen_unary(fn_name: &str, op_expr: &str) -> String {
    format!(
r##"@export("{fn_name}")
def {fn_name}(
    a_ptr: OpaquePointer[MutUntrackedOrigin],
    out_ptr: OpaquePointer[MutUntrackedOrigin],
    n: Int,
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(n):
        out.unsafe_store(i, {op_expr})"##
    )
}

/// Generate a stride-aware unary elementwise op function.
/// `op_expr` should contain the placeholder `{a}` which will be replaced with the strided index expression.
fn gen_unary_strided(
    fn_name: &str,
    op_expr: &str,
    shape: &[usize],
    a_idx_exprs: &[String],
) -> String {
    let ndim = shape.len();
    let mut loops = String::new();
    for d in 0..ndim {
        let indent = "    ".repeat(d + 1);
        loops.push_str(&format!("{indent}for i{d} in range({}):\n", shape[d]));
    }
    let inner_indent = "    ".repeat(ndim + 1);
    let flat_out = (0..ndim).map(|d| {
        let stride: usize = (d + 1..ndim).map(|dd| shape[dd]).product();
        format!("i{d} * {stride}")
    }).collect::<Vec<_>>().join(" + ");
    let a_idx = a_idx_exprs.iter()
        .map(|e| format!("({e})"))
        .collect::<Vec<_>>()
        .join(" + ");
    let op_resolved = op_expr.replace("{a}", &a_idx);

    format!(
r##"@export("{fn_name}")
def {fn_name}(
    a_ptr: OpaquePointer[MutUntrackedOrigin],
    out_ptr: OpaquePointer[MutUntrackedOrigin],
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
{loops}{inner_indent}out.unsafe_store({flat_out}, {op_resolved})"##
    )
}

/// Generate a reduction op function.
/// `acc_var` is the accumulator name, `acc_init` is the Float32 init value,
/// `acc_stmt` is the full accumulation statement (e.g. "acc += val" or "if val > best: best = val").
/// `stride_exprs` are Mojo expressions for the base index into the input, parameterized by loop variables.
/// Generate a fused GEMM kernel: out[b,m,n] = relu(a[b,m,k] @ b[b,k,n] + bias[n]).
/// Vectorized 8-wide along n (the only contiguous axis: b's rows and bias are
/// contiguous in n) with a scalar tail for n % 8; `batch` is baked into the
/// kernel body as an outermost loop, keeping the FFI signature pointer-only.
fn gen_gemm(fn_name: &str, batch: usize, m: usize, n: usize, k: usize, bias: bool, relu: bool) -> String {
    let params = if bias {
        "a_ptr: OpaquePointer[MutUntrackedOrigin],\n    b_ptr: OpaquePointer[MutUntrackedOrigin],\n    bias_ptr: OpaquePointer[MutUntrackedOrigin],\n    out_ptr: OpaquePointer[MutUntrackedOrigin]"
    } else {
        "a_ptr: OpaquePointer[MutUntrackedOrigin],\n    b_ptr: OpaquePointer[MutUntrackedOrigin],\n    out_ptr: OpaquePointer[MutUntrackedOrigin]"
    };
    let bias_cast = if bias {
        "    var bias = bias_ptr.unsafe_bitcast[Scalar[DType.float32]]()\n"
    } else {
        ""
    };

    let batched = batch > 1;
    // Indentation levels: `row` = the im loop line, `loop2` = the while/for-jn
    // lines (im-loop body), `vec_body` = their bodies + the ik loop lines +
    // epilogue statements, `ik_body` = the accumulation statements.
    let row = if batched { "        " } else { "    " };
    let loop2 = if batched { "            " } else { "        " };
    let vec_body = if batched { "                " } else { "            " };
    let ik_body = if batched { "                    " } else { "                " };
    let epi = vec_body;

    let a_batch = if batched { format!("ib * {} + ", m * k) } else { String::new() };
    let b_batch = if batched { format!("ib * {} + ", k * n) } else { String::new() };
    let batch_open = if batched {
        format!("    for ib in range({batch}):\n")
    } else {
        String::new()
    };

    let bias_vec = if bias {
        format!("{epi}acc8 = acc8 + bias.unsafe_load[width=8](jj)\n")
    } else {
        String::new()
    };
    let relu_vec = if relu {
        format!("{epi}acc8 = max(acc8, SIMD[DType.float32, 8](0))\n")
    } else {
        String::new()
    };
    let bias_scalar = if bias {
        format!("{epi}acc = acc + bias.unsafe_load(jn)\n")
    } else {
        String::new()
    };
    let relu_scalar = if relu {
        format!("{epi}if acc < Float32(0):\n{epi}    acc = Float32(0)\n")
    } else {
        String::new()
    };

    format!(
        r##"@export("{fn_name}")
def {fn_name}(
    {params}
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var b = b_ptr.unsafe_bitcast[Scalar[DType.float32]]()
{bias_cast}    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out_pos = 0
{batch_open}{row}for im in range({m}):
{loop2}var jj = 0
{loop2}while jj + 8 <= {n}:
{vec_body}var acc8 = SIMD[DType.float32, 8](0)
{vec_body}for ik in range({k}):
{ik_body}acc8 = acc8 + SIMD[DType.float32, 8](a.unsafe_load({a_batch}im * {k} + ik)) * b.unsafe_load[width=8]({b_batch}ik * {n} + jj)
{bias_vec}{relu_vec}{vec_body}out.unsafe_store[width=8](out_pos, acc8)
{vec_body}out_pos += 8
{vec_body}jj += 8
{loop2}for jn in range(jj, {n}):
{vec_body}var acc = Float32(0)
{vec_body}for ik in range({k}):
{ik_body}acc = acc + a.unsafe_load({a_batch}im * {k} + ik) * b.unsafe_load({b_batch}ik * {n} + jn)
{bias_scalar}{relu_scalar}{vec_body}out.unsafe_store(out_pos, acc)
{vec_body}out_pos += 1"##
    )
}

fn gen_reduce(
    fn_name: &str,
    acc_var: &str,
    acc_init: &str,
    acc_stmt: &str,
    out_shape: &[usize],
    stride_exprs: &[String],
    iters: usize,
    iter_stride_expr: &str,
) -> String {
    let ndim = out_shape.len();

    let mut body = String::new();
    let mut indent = "    ".to_string();

    // Outer loops: iterate over output positions
    for d in 0..ndim {
        writeln!(body, "{indent}for i{d} in range({}):", out_shape[d]).unwrap();
        indent.push_str("    ");
    }

    // Compute base index (start position in input)
    let base_terms: Vec<&str> = stride_exprs.iter().map(|s| s.as_str()).collect();
    let base_idx = if base_terms.is_empty() { "0".to_string() } else { base_terms.join(" + ") };

    writeln!(body, "{indent}var base = {base_idx}").unwrap();
    writeln!(body, "{indent}var {acc_var} = Float32({acc_init})").unwrap();
    writeln!(body, "{indent}for k in range({iters}):").unwrap();
    writeln!(body, "{indent}    var val = a.unsafe_load(base + {iter_stride_expr})").unwrap();
    writeln!(body, "{indent}    {acc_stmt}").unwrap();
    writeln!(body, "{indent}out.unsafe_store(out_pos, {acc_var})").unwrap();
    writeln!(body, "{indent}out_pos += 1").unwrap();

    format!(
r##"@export("{fn_name}")
def {fn_name}(
    a_ptr: OpaquePointer[MutUntrackedOrigin],
    out_ptr: OpaquePointer[MutUntrackedOrigin],
) abi("C") -> None:
    var a = a_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out = out_ptr.unsafe_bitcast[Scalar[DType.float32]]()
    var out_pos = 0
{body}"##
    )
}

/// Build a mapping from logical (flat) index to physical (contiguous buffer) index
/// using the shape and stride EXPRESSIONS. Each stride expression is evaluated
/// with z=index_d for dimension d, matching StridedIterator semantics.
pub fn build_strided_index_map_expr(shape: &[usize], stride_exprs: &[Expression]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 || shape.iter().product::<usize>() == 0 {
        return vec![];
    }
    let total: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total);
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        let phys: usize = stride_exprs.iter()
            .zip(&index)
            .map(|(expr, &idx)| expr.exec_single_var(idx))
            .sum();
        result.push(phys);
        // Increment multi-dimensional index (row-major / last-dim fastest)
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    result
}

/// Build a mapping from logical (flat) index to physical (contiguous buffer) index
/// using the shape and strides. Equivalent to collecting StridedIterator.
pub fn build_strided_index_map(shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 || shape.iter().product::<usize>() == 0 {
        return vec![];
    }
    let total: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total);
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        let phys: usize = strides.iter().zip(&index).map(|(s, &i)| s * i).sum();
        result.push(phys);
        // Increment multi-dimensional index (row-major / last-dim fastest)
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    result
}

/// Compute contiguous (row-major) strides for a shape.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
