//! Golden tests for a synthetic-weights Qwen-shaped transformer: the Mojo
//! backend is compared against luminal's ReferenceRuntime at several small
//! configs, plus component-level checks (attention, rmsnorm, rope, mlp,
//! shape primitives).

mod common;

use common::{
    assert_close, assert_matches_reference, compile, gen_data, max_err, run_reference_and_mojo,
};
use luminal::prelude::*;

/// Full-model outputs accumulate over layers, so the golden tolerance is
/// looser than the component-level one.
const MODEL_TOLERANCE: f32 = 0.5;
const COMPONENT_TOLERANCE: f32 = 1e-3;

struct ModelConfig {
    seq: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    n_kv_heads: usize,
    intermediate: usize,
    vocab: usize,
    n_layers: usize,
}

const TINY: ModelConfig = ModelConfig {
    seq: 2, dim: 16, n_heads: 4, head_dim: 4, n_kv_heads: 2, intermediate: 32, vocab: 32, n_layers: 2,
};
const SMALL: ModelConfig = ModelConfig {
    seq: 4, dim: 32, n_heads: 4, head_dim: 8, n_kv_heads: 2, intermediate: 64, vocab: 64, n_layers: 2,
};
const MEDIUM: ModelConfig = ModelConfig {
    seq: 4, dim: 64, n_heads: 8, head_dim: 8, n_kv_heads: 4, intermediate: 128, vocab: 128, n_layers: 4,
};

struct BuiltModel {
    cx: Graph,
    inputs: Vec<(NodeIndex, Vec<f32>)>,
    output: NodeIndex,
}

/// Single- or multi-layer Qwen-style prefill graph: gather embedding →
/// (rmsnorm → qkv → kv-cache scatter → GQA attention → o_proj → residual →
/// rmsnorm → swiglu → residual) × n_layers → tied LM head.
fn build_model(cfg: &ModelConfig) -> BuiltModel {
    let max_seq = cfg.seq * 2;
    let prev = 0;
    let total_seq = prev + cfg.seq;
    let kv_groups = cfg.n_heads / cfg.n_kv_heads;

    let mut cx = Graph::new();
    cx.set_dim('s', cfg.seq);
    cx.set_dim('p', prev);

    let embedding = cx.tensor((cfg.vocab, cfg.dim));
    let token_ids = cx.tensor((cfg.seq,));
    let mut weight_specs: Vec<(NodeIndex, usize)> = Vec::new();

    let h_offset = (token_ids.cast(DType::Int) * cfg.dim).expand_dim(1, cfg.dim);
    let d_offset = cx.arange(cfg.dim).expand_dim(0, cfg.seq);
    let mut x = embedding.gather(h_offset + d_offset);

    for _layer in 0..cfg.n_layers {
        let ms = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
        let attn_w = cx.tensor((cfg.dim,));
        weight_specs.push((attn_w.id, cfg.dim));
        let normed = x * ms.expand_dim(1, cfg.dim) * attn_w.expand_dim(0, cfg.seq);

        let q_proj = cx.tensor((cfg.n_heads * cfg.head_dim, cfg.dim));
        let k_proj = cx.tensor((cfg.n_kv_heads * cfg.head_dim, cfg.dim));
        let v_proj = cx.tensor((cfg.n_kv_heads * cfg.head_dim, cfg.dim));
        weight_specs.push((q_proj.id, cfg.n_heads * cfg.head_dim * cfg.dim));
        weight_specs.push((k_proj.id, cfg.n_kv_heads * cfg.head_dim * cfg.dim));
        weight_specs.push((v_proj.id, cfg.n_kv_heads * cfg.head_dim * cfg.dim));
        let q = normed.matmul(q_proj.t());
        let k = normed.matmul(k_proj.t());
        let v = normed.matmul(v_proj.t());

        let q_3d = q.split_dims(1, cfg.head_dim).transpose(0, 1);
        let k_3d = k.split_dims(1, cfg.head_dim).transpose(0, 1);
        let v_3d = v.split_dims(1, cfg.head_dim).transpose(0, 1);

        let k_cache = cx.tensor((cfg.n_kv_heads, max_seq, cfg.head_dim));
        let v_cache = cx.tensor((cfg.n_kv_heads, max_seq, cfg.head_dim));
        weight_specs.push((k_cache.id, cfg.n_kv_heads * max_seq * cfg.head_dim));
        weight_specs.push((v_cache.id, cfg.n_kv_heads * max_seq * cfg.head_dim));

        let kh_offset = cx.arange(cfg.n_kv_heads) * (max_seq * cfg.head_dim);
        let kp_offset = (cx.arange(cfg.seq) + prev) * cfg.head_dim;
        let kd_offset = cx.arange(cfg.head_dim);
        let scatter_idx = kh_offset
            .expand_dim(1, cfg.seq)
            .expand_dim(2, cfg.head_dim)
            + kp_offset
                .expand_dim(0, cfg.n_kv_heads)
                .expand_dim(2, cfg.head_dim)
            + kd_offset
                .expand_dim(0, cfg.n_kv_heads)
                .expand_dim(1, cfg.seq);

        let k_cache_out = k_3d.scatter(scatter_idx, k_cache);
        let v_cache_out = v_3d.scatter(scatter_idx, v_cache);

        let k_full = k_cache_out.slice((.., ..total_seq, ..));
        let v_full = v_cache_out.slice((.., ..total_seq, ..));
        let k_exp = k_full.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
        let v_exp = v_full.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;

        let scale = (cfg.head_dim as f32).sqrt().recip();
        let scores = q_3d.matmul(k_exp.transpose(1, 2)) * scale;

        let q_abs = cx.arange(cfg.seq).cast(DType::F32) + prev as f32;
        let k_pos = cx.arange(total_seq).cast(DType::F32);
        let mask = k_pos.expand_dim(0, cfg.seq).gt(q_abs.expand_dim(1, total_seq));
        let mask_3d = mask.cast(DType::F32).expand_dim(0, cfg.n_heads);
        let masked_scores = scores + mask_3d * (-1e10f32);

        let attn_weights = masked_scores.softmax(2);
        let attn_out = attn_weights.matmul(v_exp);

        let attn_flat = attn_out.transpose(0, 1).merge_dims(1, 2);
        let o_proj = cx.tensor((cfg.dim, cfg.n_heads * cfg.head_dim));
        weight_specs.push((o_proj.id, cfg.dim * cfg.n_heads * cfg.head_dim));
        let o_result = attn_flat.matmul(o_proj.t());

        x = x + o_result;

        let ms2 = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
        let mlp_w = cx.tensor((cfg.dim,));
        weight_specs.push((mlp_w.id, cfg.dim));
        let mlp_normed = x * ms2.expand_dim(1, cfg.dim) * mlp_w.expand_dim(0, cfg.seq);

        let gate_w = cx.tensor((cfg.intermediate, cfg.dim));
        let up_w = cx.tensor((cfg.intermediate, cfg.dim));
        let down_w = cx.tensor((cfg.dim, cfg.intermediate));
        weight_specs.push((gate_w.id, cfg.intermediate * cfg.dim));
        weight_specs.push((up_w.id, cfg.intermediate * cfg.dim));
        weight_specs.push((down_w.id, cfg.dim * cfg.intermediate));
        let gate = mlp_normed.matmul(gate_w.t());
        let up = mlp_normed.matmul(up_w.t());
        let act = gate.swish() * up;
        let mlp_out = act.matmul(down_w.t());

        x = x + mlp_out;
    }

    let logits = x.matmul(embedding.t());
    let out = logits.output();

    let mut inputs = vec![
        (
            embedding.id,
            gen_data(cfg.vocab * cfg.dim, 0.01),
        ),
        (
            token_ids.id,
            (0..cfg.seq).map(|i| (i * 3 + 1) as f32).collect(),
        ),
    ];
    for (id, n) in &weight_specs {
        inputs.push((*id, gen_data(*n, 0.1)));
    }

    BuiltModel { cx, inputs, output: out.id }
}

fn golden_check(cfg: &ModelConfig, label: &str) {
    let mut bg_ref = build_model(cfg);
    bg_ref
        .cx
        .build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt_ref = ReferenceRuntime::default();
    rt_ref = bg_ref.cx.search(rt_ref, CompileOptions::default());
    for (id, data) in &bg_ref.inputs {
        rt_ref.set_data(*id, data.clone());
    }
    rt_ref.execute(&bg_ref.cx.dyn_map);
    let ref_out = rt_ref.get_f32(bg_ref.output).clone();

    let mut bg_mojo = build_model(cfg);
    let mut rt_mojo = compile(&mut bg_mojo.cx, CompileOptions::default());
    for (id, data) in &bg_mojo.inputs {
        rt_mojo.set_data_f32(*id, data);
    }
    rt_mojo.execute(&bg_mojo.cx.dyn_map);
    let mojo_out = rt_mojo.get_f32(bg_mojo.output);

    let err = max_err(&ref_out, &mojo_out);
    assert!(
        err < MODEL_TOLERANCE,
        "{label}: max error {err} exceeds tolerance {MODEL_TOLERANCE}"
    );
}

#[test]
fn golden_tiny_model_matches_reference() {
    golden_check(&TINY, "tiny");
}

#[test]
fn golden_small_model_matches_reference() {
    golden_check(&SMALL, "small");
}

#[test]
fn golden_medium_model_matches_reference() {
    golden_check(&MEDIUM, "medium");
}

#[test]
fn token_embedding_gather_matches_reference() {
    let (seq, dim, vocab) = (4usize, 16usize, 32usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            cx.set_dim('s', seq);
            let embedding = cx.tensor((vocab, dim));
            let token_ids = cx.tensor((seq,));
            let h_offset = (token_ids.cast(DType::Int) * dim).expand_dim(1, dim);
            let d_offset = cx.arange(dim).expand_dim(0, seq);
            let out = embedding.gather(h_offset + d_offset).output();
            (
                vec![
                    (embedding.id, gen_data(vocab * dim, 0.01)),
                    (token_ids.id, vec![1.0, 5.0, 3.0, 7.0]),
                ],
                out.id,
            )
        },
        1e-5,
        "embedding gather",
    );
}

#[test]
fn rmsnorm_matches_host_computation() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 4));
    let weight = cx.tensor((4,));
    let eps = 1e-6f32;

    let ms = ((x * x).mean(1) + eps).sqrt().reciprocal();
    let normed = x * ms.expand_dim(1, 4) * weight.expand_dim(0, 2);
    let out = normed.output();

    let x_data: Vec<f32> = vec![3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let w_data: Vec<f32> = vec![2.0, 3.0, 0.5, 1.0];

    let mut expected = vec![0.0f32; 8];
    for row in 0..2 {
        let sq_sum: f32 = (0..4).map(|c| x_data[row * 4 + c].powi(2)).sum();
        let rms = ((sq_sum / 4.0) + eps).sqrt().recip();
        for c in 0..4 {
            expected[row * 4 + c] = x_data[row * 4 + c] * rms * w_data[c];
        }
    }

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &x_data);
    rt.set_data_f32(weight.id, &w_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-3, "rmsnorm");
}

#[test]
fn softmax_matches_host_computation() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 4));
    let out = x.softmax(1).output();

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
    let mut expected = vec![0.0f32; 8];
    for row in 0..2 {
        let max_val = x_data[row * 4..row * 4 + 4]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = x_data[row * 4..row * 4 + 4]
            .iter()
            .map(|v| (v - max_val).exp())
            .collect();
        let sum: f32 = exp_vals.iter().sum();
        for c in 0..4 {
            expected[row * 4 + c] = exp_vals[c] / sum;
        }
    }

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &x_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-3, "softmax");
}

#[test]
fn swish_matches_host_computation() {
    let mut cx = Graph::new();
    let x = cx.tensor(4);
    let out = x.swish().output();

    let x_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0];
    let expected: Vec<f32> = x_data
        .iter()
        .map(|&v| {
            let sig = 1.0 / (1.0 + (-v).exp());
            v * sig
        })
        .collect();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &x_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-3, "swish");
}

#[test]
fn single_head_attention_matches_host_computation() {
    let mut cx = Graph::new();
    let q = cx.tensor((2, 4));
    let k = cx.tensor((2, 4));
    let v = cx.tensor((2, 4));

    let scale = (4.0f32).sqrt().recip();
    let scores = q.matmul(k.transpose(0, 1)) * scale;
    let attn = scores.softmax(1);
    let out = attn.matmul(v).output();

    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let k_data = q_data.clone();
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let s00: f32 = 1.0 * scale;
    let s11: f32 = 1.0 * scale;
    let ez0: f32 = s00.exp();
    let ez1: f32 = s11.exp();
    let p00: f32 = s00.exp() / ez0;
    let p11: f32 = s11.exp() / ez1;
    let mut expected = vec![0.0f32; 8];
    for c in 0..4 {
        expected[c] = p00 * v_data[c] + (1.0 - p00) * v_data[4 + c];
        expected[4 + c] = (1.0 - p11) * v_data[c] + p11 * v_data[4 + c];
    }

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(q.id, &q_data);
    rt.set_data_f32(k.id, &k_data);
    rt.set_data_f32(v.id, &v_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-3, "attention");
}

#[test]
fn slice_concat_roundtrip() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 8));
    let x0 = x.slice((.., ..4));
    let x1 = x.slice((.., 4..));
    let out = x0.concat_along(x1, 1).output();

    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &data, 1e-4, "slice_concat");
}

#[test]
fn pad_then_add_concatenation() {
    let mut cx = Graph::new();
    let a = cx.tensor(3);
    let b = cx.tensor(3);
    let out = (a.pad((0, 3), 0.0) + b.pad((3, 0), 0.0)).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, &[1.0, 2.0, 3.0]);
    rt.set_data_f32(b.id, &[4.0, 5.0, 6.0]);
    rt.execute(&cx.dyn_map);

    assert_close(
        &rt.get_f32(out.id),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-4,
        "pad_add",
    );
}

#[test]
fn split_merge_dims_roundtrip() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 8));
    let out = x.split_dims(1, 4).merge_dims(1, 2).output();

    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &data, 1e-4, "split_merge");
}

#[test]
fn transpose_2d() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 3));
    let out = x.transpose(0, 1).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(x.id, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt.execute(&cx.dyn_map);

    assert_close(
        &rt.get_f32(out.id),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        1e-4,
        "transpose",
    );
}

#[test]
fn swiglu_mlp_matches_reference() {
    assert_matches_reference(
        |cx: &mut Graph| {
            let x = cx.tensor((1, 4));
            let gate_w = cx.tensor((8, 4));
            let up_w = cx.tensor((8, 4));
            let down_w = cx.tensor((4, 8));
            let gate = x.matmul(gate_w.t());
            let up = x.matmul(up_w.t());
            let act = gate.swish() * up;
            let out = act.matmul(down_w.t()).output();
            (
                vec![
                    (x.id, vec![1.0, 2.0, 3.0, 4.0]),
                    (gate_w.id, gen_data(32, 0.1)),
                    (up_w.id, gen_data(32, 0.2)),
                    (down_w.id, gen_data(32, 0.3)),
                ],
                out.id,
            )
        },
        COMPONENT_TOLERANCE,
        "swiglu mlp",
    );
}

#[test]
fn rope_matches_reference() {
    let (n_heads, head_dim) = (2usize, 4usize);
    let (ref_out, mojo_out) = run_reference_and_mojo(|cx: &mut Graph| {
        let input = cx.tensor((2, n_heads * head_dim));
        let pos_ids = cx.tensor((2,));

        let x = input.split_dims(1, head_dim).transpose(0, 1);
        let freqs = cx.arange_options(0, head_dim, 2).cast(DType::F32) / head_dim as f32;
        let inv_freqs = 1_000_000f32.pow(freqs).reciprocal();
        let emb = pos_ids
            .cast(DType::F32)
            .expand_dim(1, 1)
            .matmul(inv_freqs.expand_dim(0, 1));

        let x0 = x.slice((.., .., ..head_dim / 2));
        let x1 = x.slice((.., .., head_dim / 2..));
        let cos = emb.cos().expand_dim(0, n_heads);
        let sin = emb.sin().expand_dim(0, n_heads);
        let x0_out = x0 * cos - x1 * sin;
        let x1_out = x1 * cos + x0 * sin;

        let out = x0_out
            .concat_along(x1_out, 2)
            .transpose(0, 1)
            .merge_dims(1, 2)
            .output();
        (
            vec![
                (
                    input.id,
                    (1..=n_heads * head_dim * 2).map(|i| i as f32).collect(),
                ),
                (pos_ids.id, vec![0.0, 1.0]),
            ],
            out.id,
        )
    });

    assert_close(&mojo_out, &ref_out, COMPONENT_TOLERANCE, "rope");
    // Position 0 has cos=1, sin=0, so the first sequence element is unchanged.
    for (i, &v) in mojo_out[..n_heads * head_dim].iter().enumerate() {
        assert!(
            (v - (i as f32 + 1.0)).abs() < 1e-3,
            "rope pos 0 [{i}]: got {v}, expected {}",
            i as f32 + 1.0
        );
    }
}

#[test]
fn multi_head_attention_matches_reference() {
    let (n_heads, head_dim, seq) = (2usize, 4usize, 2usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let q = cx.tensor((seq, n_heads * head_dim));
            let k = cx.tensor((seq, n_heads * head_dim));
            let v = cx.tensor((seq, n_heads * head_dim));

            let q_3d = q.split_dims(1, head_dim).transpose(0, 1);
            let k_3d = k.split_dims(1, head_dim).transpose(0, 1);
            let v_3d = v.split_dims(1, head_dim).transpose(0, 1);

            let scale = (head_dim as f32).sqrt().recip();
            let scores = q_3d.matmul(k_3d.transpose(1, 2)) * scale;
            let attn_out = scores.softmax(2).matmul(v_3d);
            let out = attn_out.transpose(0, 1).merge_dims(1, 2).output();

            let q_data: Vec<f32> = vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];
            (
                vec![
                    (q.id, q_data.clone()),
                    (k.id, q_data),
                    (
                        v.id,
                        (1..=n_heads * head_dim * seq).map(|i| i as f32).collect(),
                    ),
                ],
                out.id,
            )
        },
        COMPONENT_TOLERANCE,
        "multi-head attention",
    );
}

#[test]
fn causal_attention_with_kv_cache_matches_reference() {
    let (n_heads, n_kv_heads, head_dim, max_seq, prev, seq) =
        (2usize, 1usize, 4usize, 8usize, 2usize, 2usize);
    let kv_groups = n_heads / n_kv_heads;
    let total_seq = prev + seq;
    assert_matches_reference(
        |cx: &mut Graph| {
            cx.set_dim('s', seq);
            cx.set_dim('p', prev);
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;

            let q = cx.tensor((seq, q_dim));
            let k = cx.tensor((seq, kv_dim));
            let v = cx.tensor((seq, kv_dim));
            let k_cache = cx.tensor((n_kv_heads, max_seq, head_dim));
            let v_cache = cx.tensor((n_kv_heads, max_seq, head_dim));

            let q_3d = q.split_dims(1, head_dim).transpose(0, 1);
            let k_new = k.split_dims(1, head_dim).transpose(0, 1);
            let v_new = v.split_dims(1, head_dim).transpose(0, 1);

            let h_offset = cx.arange(n_kv_heads) * (max_seq * head_dim);
            let p_offset = (cx.arange(seq) + prev) * head_dim;
            let d_offset = cx.arange(head_dim);
            let scatter_idx = h_offset
                .expand_dim(1, seq)
                .expand_dim(2, head_dim)
                + p_offset
                    .expand_dim(0, n_kv_heads)
                    .expand_dim(2, head_dim)
                + d_offset
                    .expand_dim(0, n_kv_heads)
                    .expand_dim(1, seq);

            let k_full = k_new.scatter(scatter_idx, k_cache).slice((.., ..total_seq, ..));
            let v_full = v_new.scatter(scatter_idx, v_cache).slice((.., ..total_seq, ..));
            let k_exp = k_full.expand_dim(1, kv_groups).merge_dims(0, 1);
            let v_exp = v_full.expand_dim(1, kv_groups).merge_dims(0, 1);

            let scale = (head_dim as f32).sqrt().recip();
            let scores = q_3d.matmul(k_exp.transpose(1, 2)) * scale;

            let q_abs = cx.arange(seq).cast(DType::F32) + prev as f32;
            let k_pos = cx.arange(total_seq).cast(DType::F32);
            let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total_seq));
            let masked = scores + mask.cast(DType::F32).expand_dim(0, n_heads) * (-1e10f32);

            let out = masked
                .softmax(2)
                .matmul(v_exp)
                .transpose(0, 1)
                .merge_dims(1, 2)
                .output();

            (
                vec![
                    (q.id, (1..=seq * q_dim).map(|i| i as f32).collect()),
                    (k.id, (1..=seq * kv_dim).map(|i| i as f32 * 0.1).collect()),
                    (v.id, (1..=seq * kv_dim).map(|i| i as f32 * 0.1).collect()),
                    (k_cache.id, vec![0.0; n_kv_heads * max_seq * head_dim]),
                    (v_cache.id, vec![0.0; n_kv_heads * max_seq * head_dim]),
                ],
                out.id,
            )
        },
        COMPONENT_TOLERANCE,
        "kv-cache causal attention",
    );
}

#[test]
fn transformer_block_matches_reference() {
    let (seq, dim, n_heads, head_dim, n_kv_heads, intermediate) =
        (2usize, 8usize, 2usize, 4usize, 2usize, 16usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let x = cx.tensor((seq, dim));

            let attn_w = cx.tensor((dim,));
            let ms = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
            let normed = x * ms.expand_dim(1, dim) * attn_w.expand_dim(0, seq);

            let q_proj = cx.tensor((n_heads * head_dim, dim));
            let k_proj = cx.tensor((n_kv_heads * head_dim, dim));
            let v_proj = cx.tensor((n_kv_heads * head_dim, dim));
            let q = normed.matmul(q_proj.t());
            let k = normed.matmul(k_proj.t());
            let v = normed.matmul(v_proj.t());

            let q_3d = q.split_dims(1, head_dim).transpose(0, 1);
            let k_3d = k.split_dims(1, head_dim).transpose(0, 1);
            let v_3d = v.split_dims(1, head_dim).transpose(0, 1);

            let scale = (head_dim as f32).sqrt().recip();
            let scores = q_3d.matmul(k_3d.transpose(1, 2)) * scale;
            let attn_out = scores.softmax(2).matmul(v_3d);

            let attn_flat = attn_out.transpose(0, 1).merge_dims(1, 2);
            let o_proj = cx.tensor((dim, n_heads * head_dim));
            let after_attn = x + attn_flat.matmul(o_proj.t());

            let mlp_w = cx.tensor((dim,));
            let ms2 = ((after_attn * after_attn).mean(1) + 1e-6f32).sqrt().reciprocal();
            let mlp_normed = after_attn * ms2.expand_dim(1, dim) * mlp_w.expand_dim(0, seq);

            let gate_w = cx.tensor((intermediate, dim));
            let up_w = cx.tensor((intermediate, dim));
            let down_w = cx.tensor((dim, intermediate));
            let act = mlp_normed.matmul(gate_w.t()).swish() * mlp_normed.matmul(up_w.t());
            let out = (after_attn + act.matmul(down_w.t())).output();

            let lin = |n: usize, seed: f32| gen_data(n, seed);
            (
                vec![
                    (x.id, vec![0.5; seq * dim]),
                    (attn_w.id, vec![1.0; dim]),
                    (q_proj.id, lin(n_heads * head_dim * dim, 0.1)),
                    (k_proj.id, lin(n_kv_heads * head_dim * dim, 0.2)),
                    (v_proj.id, lin(n_kv_heads * head_dim * dim, 0.3)),
                    (o_proj.id, lin(dim * n_heads * head_dim, 0.4)),
                    (mlp_w.id, vec![1.0; dim]),
                    (gate_w.id, lin(intermediate * dim, 0.5)),
                    (up_w.id, lin(intermediate * dim, 0.6)),
                    (down_w.id, lin(dim * intermediate, 0.7)),
                ],
                out.id,
            )
        },
        COMPONENT_TOLERANCE,
        "transformer block",
    );
}
