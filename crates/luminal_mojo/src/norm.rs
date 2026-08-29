//! Mojo RMSNorm backend op: e-graph rewrites that lower the row-wise RMSNorm
//! pattern into a single fused `MojoRMSNorm` kernel
//! (`out[r,c] = x[r,c] * rsqrt(mean(x[r,:]²) + eps)`), mirroring the fused
//! GEMM backend's rewrite structure (`gemm.rs`).

use luminal::dtype::DType;
use luminal::egglog_utils::{
    api::{Rule, SortDef, sort},
    base::{DTYPE, EXPRESSION, F64, OP_KIND},
    extract_dtype, extract_expr, SerializedEGraph,
};
use luminal::op::{EgglogOp, LLIROp};
use luminal::prelude::*;
use luminal::shape::Expression;

use crate::gemm::MojoOp;

/// `out[r,c] = x[r,c] * (mean(x[r,:]²) + eps)^-0.5`, row-wise over the last axis.
#[derive(Debug)]
pub struct MojoRMSNormLLIR {
    pub rows: Expression,
    pub cols: Expression,
    pub eps: f32,
    pub dtype: DType,
}

impl MojoOp for MojoRMSNormLLIR {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Egglog registration for the `MojoRMSNorm` op kind. The unit instance carries
/// the sort + rules; enode payloads are (rows, cols, eps, dtype).
#[derive(Debug, Default, Clone)]
pub struct MojoRMSNorm;

impl EgglogOp for MojoRMSNorm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "MojoRMSNorm",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("eps", F64),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec!["(relation mojo_rmsnorm_base_dtype (DType))
     (mojo_rmsnorm_base_dtype (F32))"
            .to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("mojo_rmsnorm_rewrite.egg"))]
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
        let rows = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let cols = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let eps = egraph.enodes[kind_children[2]]
            .0
            .replace("\"", "")
            .parse::<f32>()
            .unwrap();
        let dtype = extract_dtype(egraph, kind_children[3]);

        let extracted = MojoRMSNormLLIR {
            rows,
            cols,
            eps,
            dtype,
        };
        (
            LLIROp::new::<dyn MojoOp>(Box::new(extracted) as Box<dyn MojoOp>),
            input_enodes,
        )
    }
}

/// `out[r,c] = exp2((x[r,c] + c1·max(x[r,:]))·c2) / Σ_c exp2((x[r,c] + c1·max(x[r,:]))·c2)`,
/// row-wise over the last axis. The canonical frontend softmax lowering
/// (`x - max` → `exp` → `/ sum`) instantiates this with c1 = -1 and
/// c2 = 1/ln(2); the constants are bound generically and baked into the kernel.
#[derive(Debug)]
pub struct MojoSoftmaxLLIR {
    pub rows: Expression,
    pub cols: Expression,
    pub c1: f32,
    pub c2: f32,
    pub dtype: DType,
}

impl MojoOp for MojoSoftmaxLLIR {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Egglog registration for the `MojoSoftmax` op kind. The unit instance carries
/// the sort + rules; enode payloads are (rows, cols, c1, c2, dtype).
#[derive(Debug, Default, Clone)]
pub struct MojoSoftmax;

impl EgglogOp for MojoSoftmax {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "MojoSoftmax",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("c1", F64),
                ("c2", F64),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec!["(relation mojo_softmax_base_dtype (DType))
     (mojo_softmax_base_dtype (F32))"
            .to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("mojo_softmax_rewrite.egg"))]
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
        let rows = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let cols = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let c1 = egraph.enodes[kind_children[2]]
            .0
            .replace("\"", "")
            .parse::<f32>()
            .unwrap();
        let c2 = egraph.enodes[kind_children[3]]
            .0
            .replace("\"", "")
            .parse::<f32>()
            .unwrap();
        let dtype = extract_dtype(egraph, kind_children[4]);

        let extracted = MojoSoftmaxLLIR {
            rows,
            cols,
            c1,
            c2,
            dtype,
        };
        (
            LLIROp::new::<dyn MojoOp>(Box::new(extracted) as Box<dyn MojoOp>),
            input_enodes,
        )
    }
}
