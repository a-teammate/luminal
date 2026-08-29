//! Mojo GEMM backend op: e-graph rewrites that lower row-major matmuls into a
//! single fused `MojoGemm` kernel (accumulation loop + bias/relu epilogues),
//! mirroring the cuBLASLt backend's rewrite structure
//! (`luminal_cuda_lite/src/host/cublaslt/`).

use luminal::dtype::DType;
use luminal::egglog_utils::{
    api::{Rule, SortDef, sort},
    base::{DTYPE, EXPRESSION, OP_KIND, STRING},
    extract_dtype, extract_expr, SerializedEGraph,
};
use luminal::op::{EgglogOp, LLIROp};
use luminal::prelude::*;
use luminal::shape::Expression;

/// LLIR dialect trait for ops emitted by the Mojo backend's egglog rules.
pub trait MojoOp: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

/// `out[m,n] = relu(a[m,k] @ b[k,n] + bias[n])`, with bias/relu optional.
#[derive(Debug)]
pub struct MojoGemmLLIR {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub batch: Expression,
    /// A's batch-dim index map (over the batch index).
    pub a_batch: Expression,
    /// A's row (m) index map.
    pub a_row: Expression,
    /// B's batch-dim index map (may contain the GQA group div: (z/g)*stride).
    pub b_batch: Expression,
    /// B's k index map.
    pub b_k: Expression,
    /// B's n index map.
    pub b_n: Expression,
    pub bias: bool,
    pub relu: bool,
    pub dtype: DType,
}

impl MojoOp for MojoGemmLLIR {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Egglog registration for the `MojoGemm` op kind. The unit instance carries
/// the sort + rules; enode payloads are (m, n, k, bias, act, dtype).
#[derive(Debug, Default, Clone)]
pub struct MojoGemm;

impl EgglogOp for MojoGemm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "MojoGemm",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("batch", EXPRESSION),
                ("a_batch", EXPRESSION),
                ("a_row", EXPRESSION),
                ("b_batch", EXPRESSION),
                ("b_k", EXPRESSION),
                ("b_n", EXPRESSION),
                ("bias", STRING),
                ("act", STRING),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2 // default instance has no bias; fused enodes carry 3
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec!["(relation mojo_gemm_base_dtype (DType))
     (mojo_gemm_base_dtype (F32))"
            .to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("mojo_gemm_rewrite.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let m = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let n = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let batch = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let a_batch = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        let a_row = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let b_batch = extract_expr(egraph, kind_children[6], expr_cache).unwrap();
        let b_k = extract_expr(egraph, kind_children[7], expr_cache).unwrap();
        let b_n = extract_expr(egraph, kind_children[8], expr_cache).unwrap();
        let bias = egraph.enodes[kind_children[9]].0.trim_matches('"') == "bias";
        let relu = egraph.enodes[kind_children[10]].0.trim_matches('"') == "relu";
        let dtype = extract_dtype(egraph, kind_children[11]);

        let extracted = MojoGemmLLIR {
            m,
            n,
            k,
            batch,
            a_batch,
            a_row,
            b_batch,
            b_k,
            b_n,
            bias,
            relu,
            dtype,
        };
        (
            LLIROp::new::<dyn MojoOp>(Box::new(extracted) as Box<dyn MojoOp>),
            input_enodes,
        )
    }
}

