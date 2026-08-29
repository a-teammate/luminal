//! Consolidated benchmark: MojoRuntime vs ReferenceRuntime execution time on
//! a matmul pipeline and a transformer block at several sizes. Numerics are
//! verified once per configuration; the timing loops do not re-check.
//!
//! Run with: cargo bench -p luminal_mojo

#[path = "../tests/common/mod.rs"]
mod common;

use common::{compile, gen_data, max_err};
use luminal::prelude::*;
use std::time::Instant;

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;

/// Mean wall-clock milliseconds per `execute` call.
fn time_execution<R: Runtime>(rt: &mut R, dyn_map: &FxHashMap<char, usize>) -> f64 {
    for _ in 0..WARMUP_ITERS {
        rt.execute(dyn_map);
    }
    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        rt.execute(dyn_map);
    }
    t0.elapsed().as_secs_f64() * 1e3 / TIMED_ITERS as f64
}

fn build_matmul_pipeline(cx: &mut Graph, size: usize) -> (Vec<(NodeIndex, Vec<f32>)>, NodeIndex) {
    let a = cx.tensor((size, size));
    let b = cx.tensor((size, size));
    let w = cx.tensor((size,));

    let c = a.matmul(b);
    let d = c.swish();
    let ms = ((d * d).mean(1) + 1e-6f32).sqrt().reciprocal();
    let e = d * ms.expand_dim(1, size) * w.expand_dim(0, size);
    let f = e.matmul(b);
    let out = f.output();

    let inputs = vec![
        (a.id, gen_data(size * size, 1.0)),
        (b.id, gen_data(size * size, 2.0)),
        (w.id, gen_data(size, 3.0)),
    ];
    (inputs, out.id)
}

#[allow(clippy::too_many_arguments)]
fn build_transformer_block(
    cx: &mut Graph,
    seq: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
    n_kv_heads: usize,
) -> (Vec<(NodeIndex, Vec<f32>)>, NodeIndex) {
    let kv_groups = n_heads / n_kv_heads;
    let intermediate = dim * 4;

    let x = cx.tensor((seq, dim));
    let attn_w = cx.tensor((dim,));
    let q_proj = cx.tensor((n_heads * head_dim, dim));
    let k_proj = cx.tensor((n_kv_heads * head_dim, dim));
    let v_proj = cx.tensor((n_kv_heads * head_dim, dim));
    let o_proj = cx.tensor((dim, n_heads * head_dim));
    let mlp_w = cx.tensor((dim,));
    let gate_w = cx.tensor((intermediate, dim));
    let up_w = cx.tensor((intermediate, dim));
    let down_w = cx.tensor((dim, intermediate));

    let ms = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
    let normed = x * ms.expand_dim(1, dim) * attn_w.expand_dim(0, seq);

    let q = normed.matmul(q_proj.t());
    let k = normed.matmul(k_proj.t());
    let v = normed.matmul(v_proj.t());

    let q_3d = q.split_dims(1, head_dim).transpose(0, 1);
    let k_3d = k.split_dims(1, head_dim).transpose(0, 1);
    let v_3d = v.split_dims(1, head_dim).transpose(0, 1);

    let k_exp = k_3d.expand_dim(1, kv_groups).merge_dims(0, 1);
    let v_exp = v_3d.expand_dim(1, kv_groups).merge_dims(0, 1);

    let scale = (head_dim as f32).sqrt().recip();
    let scores = q_3d.matmul(k_exp.transpose(1, 2)) * scale;

    let q_pos = cx.arange(seq).cast(DType::F32);
    let k_pos = cx.arange(seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_pos.expand_dim(1, seq));
    let masked_scores = scores + mask.cast(DType::F32).expand_dim(0, n_heads) * (-1e10f32);

    let attn_out = masked_scores.softmax(2).matmul(v_exp);

    let attn_flat = attn_out.transpose(0, 1).merge_dims(1, 2);
    let x2 = x + attn_flat.matmul(o_proj.t());

    let ms2 = ((x2 * x2).mean(1) + 1e-6f32).sqrt().reciprocal();
    let mlp_normed = x2 * ms2.expand_dim(1, dim) * mlp_w.expand_dim(0, seq);

    let gate = mlp_normed.matmul(gate_w.t());
    let up = mlp_normed.matmul(up_w.t());
    let act = gate.swish() * up;
    let final_out = (x2 + act.matmul(down_w.t())).output();

    let inputs = vec![
        (x.id, gen_data(seq * dim, 0.1)),
        (attn_w.id, gen_data(dim, 0.2)),
        (q_proj.id, gen_data(n_heads * head_dim * dim, 0.3)),
        (k_proj.id, gen_data(n_kv_heads * head_dim * dim, 0.4)),
        (v_proj.id, gen_data(n_kv_heads * head_dim * dim, 0.5)),
        (o_proj.id, gen_data(dim * n_heads * head_dim, 0.6)),
        (mlp_w.id, gen_data(dim, 0.7)),
        (gate_w.id, gen_data(intermediate * dim, 0.8)),
        (up_w.id, gen_data(intermediate * dim, 0.9)),
        (down_w.id, gen_data(dim * intermediate, 1.0)),
    ];

    (inputs, final_out.id)
}

fn main() {
    println!("MojoRuntime vs ReferenceRuntime (mean ms/execute over {TIMED_ITERS} iters)");
    bench_matmul_pipeline();
    bench_transformer_block();
}

fn bench_matmul_pipeline() {
    println!();
    println!("--- matmul pipeline (matmul → swish → rmsnorm → matmul) ---");
    println!(
        "{:<10} {:>10} {:>10} {:>9} {:>12}",
        "size", "ref ms", "mojo ms", "speedup", "max err"
    );

    for size in [32usize, 64, 128] {
        let mut cx_ref = Graph::new();
        let (inputs, out) = build_matmul_pipeline(&mut cx_ref, size);
        cx_ref
            .build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt_ref = ReferenceRuntime::default();
        rt_ref = cx_ref.search(rt_ref, CompileOptions::default());
        for (id, data) in &inputs {
            rt_ref.set_data(*id, data.clone());
        }
        let ref_out = rt_ref.get_f32(out).clone();
        let ref_ms = time_execution(&mut rt_ref, &cx_ref.dyn_map);

        let mut cx_mj = Graph::new();
        let (inputs, out) = build_matmul_pipeline(&mut cx_mj, size);
        let mut rt_mj = compile(&mut cx_mj, CompileOptions::default());
        for (id, data) in &inputs {
            rt_mj.set_data_f32(*id, data);
        }
        let mojo_out = rt_mj.get_f32(out);
        let mojo_ms = time_execution(&mut rt_mj, &cx_mj.dyn_map);

        let err = max_err(&ref_out, &mojo_out);
        assert!(
            err < 1e-2,
            "matmul pipeline {size}: max error {err} exceeds 1e-2"
        );
        println!(
            "{:<10} {:>10.3} {:>10.3} {:>8.1}x {:>12.2e}",
            format!("{size}x{size}"),
            ref_ms,
            mojo_ms,
            ref_ms / mojo_ms.max(1e-6),
            err
        );
    }
}

fn bench_transformer_block() {
    println!();
    println!("--- transformer block (gqa attention + swiglu mlp) ---");
    println!(
        "{:<10} {:<20} {:>10} {:>10} {:>9} {:>12}",
        "config", "dims", "ref ms", "mojo ms", "speedup", "max err"
    );

    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("tiny", 2, 16, 4, 4, 2),
        ("small", 4, 32, 4, 8, 2),
        ("medium", 4, 64, 8, 8, 4),
        ("large", 8, 128, 16, 8, 8),
    ];

    for &(name, seq, dim, nh, hd, nkh) in configs {
        let mut cx_ref = Graph::new();
        cx_ref.set_dim('s', seq);
        let (inputs, out) = build_transformer_block(&mut cx_ref, seq, dim, nh, hd, nkh);
        cx_ref
            .build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt_ref = ReferenceRuntime::default();
        rt_ref = cx_ref.search(rt_ref, CompileOptions::default());
        for (id, data) in &inputs {
            rt_ref.set_data(*id, data.clone());
        }
        let ref_out = rt_ref.get_f32(out).clone();
        let ref_ms = time_execution(&mut rt_ref, &cx_ref.dyn_map);

        let mut cx_mj = Graph::new();
        cx_mj.set_dim('s', seq);
        let (inputs, out) = build_transformer_block(&mut cx_mj, seq, dim, nh, hd, nkh);
        let mut rt_mj = compile(&mut cx_mj, CompileOptions::default());
        for (id, data) in &inputs {
            rt_mj.set_data_f32(*id, data);
        }
        let mojo_out = rt_mj.get_f32(out);
        let mojo_ms = time_execution(&mut rt_mj, &cx_mj.dyn_map);

        let err = max_err(&ref_out, &mojo_out);
        assert!(
            err < 0.5,
            "transformer block '{name}': max error {err} exceeds 0.5"
        );
        println!(
            "{:<10} {:<20} {:>10.3} {:>10.3} {:>8.1}x {:>12.2e}",
            name,
            format!("s{seq} d{dim} h{nh} hd{hd} kv{nkh}"),
            ref_ms,
            mojo_ms,
            ref_ms / mojo_ms.max(1e-6),
            err
        );
    }
}
