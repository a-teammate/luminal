//! Per-op coverage for the Mojo backend. Each test drives one op or codegen
//! path through the standard compile pipeline and checks the output against a
//! host-computed expectation or the reference runtime.

mod common;

use common::{assert_close, assert_matches_reference, compile, gen_data};
use luminal::prelude::*;

fn run_unary(data: &[f32], op: impl FnOnce(GraphTensor) -> GraphTensor) -> Vec<f32> {
    let mut cx = Graph::new();
    let a = cx.tensor(data.len());
    let out = op(a).output();
    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, data);
    rt.execute(&cx.dyn_map);
    rt.get_f32(out.id)
}

fn run_binary(
    a_data: &[f32],
    b_data: &[f32],
    op: impl FnOnce(GraphTensor, GraphTensor) -> GraphTensor,
) -> Vec<f32> {
    let mut cx = Graph::new();
    let a = cx.tensor(a_data.len());
    let b = cx.tensor(b_data.len());
    let out = op(a, b).output();
    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, a_data);
    rt.set_data_f32(b.id, b_data);
    rt.execute(&cx.dyn_map);
    rt.get_f32(out.id)
}

#[test]
fn add_contiguous() {
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let b = cx.tensor((4, 4));
    let c = (a + b).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    let data_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    rt.set_data_f32(a.id, &data_a);
    rt.set_data_f32(b.id, &data_b);
    rt.execute(&cx.dyn_map);

    let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(x, y)| x + y).collect();
    assert_close(&rt.get_f32(c.id), &expected, 1e-6, "add");
}

#[test]
fn mul_contiguous() {
    let result = run_binary(
        &(0..16).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
        &(0..16).map(|i| (i + 1) as f32 * 2.0).collect::<Vec<_>>(),
        |a, b| a * b,
    );
    let expected: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * ((i + 1) as f32 * 2.0)).collect();
    assert_close(&result, &expected, 1e-6, "mul");
}

#[test]
fn mul_strided_broadcast() {
    // a[b,m,k] * w[b,n,k] with both operands expanded: a over n, w over m.
    let (b, m, n, k) = (2usize, 2usize, 2usize, 3usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let a = cx.tensor((b, m, k));
            let w = cx.tensor((b, n, k));
            let out = (a.expand_dim(2, n) * w.expand_dim(1, m)).output();
            (
                vec![
                    (a.id, gen_data(b * m * k, 0.1)),
                    (w.id, gen_data(b * n * k, 0.2)),
                ],
                out.id,
            )
        },
        1e-5,
        "broadcast multiply",
    );
}

#[test]
fn less_than_comparison() {
    let result = run_binary(
        &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
        &[2.0, 5.0, 4.0, 3.0, 2.0, 7.0],
        |a, b| a.lt(b),
    );
    assert_close(&result, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 1e-4, "less_than");
}

#[test]
fn exp2_unary() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5, 10.0, -2.0];
    let expected: Vec<f32> = data.iter().map(|x| x.exp2()).collect();
    assert_close(&run_unary(&data, |a| a.exp2()), &expected, 1e-5, "exp2");
}

#[test]
fn log2_unary() {
    let data: Vec<f32> = vec![1.0, 2.0, 4.0, 8.0, 0.5, 16.0, 3.0, 7.0];
    let expected: Vec<f32> = data.iter().map(|x| x.log2()).collect();
    assert_close(&run_unary(&data, |a| a.log2()), &expected, 1e-5, "log2");
}

#[test]
fn sin_unary() {
    let data: Vec<f32> = vec![-1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0, -3.0];
    let expected: Vec<f32> = data.iter().map(|x| x.sin()).collect();
    assert_close(&run_unary(&data, |a| a.sin()), &expected, 1e-5, "sin");
}

#[test]
fn reciprocal_unary() {
    let data: Vec<f32> = vec![1.0, 2.0, 4.0, 0.5, -1.0, 8.0, -2.0, 0.25];
    let expected: Vec<f32> = data.iter().map(|x| 1.0 / x).collect();
    assert_close(&run_unary(&data, |a| a.reciprocal()), &expected, 1e-5, "reciprocal");
}

#[test]
fn sqrt_unary() {
    let data: Vec<f32> = vec![0.0, 1.0, 4.0, 9.0, 0.25, 16.0, 2.25, 100.0];
    let expected: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
    assert_close(&run_unary(&data, |a| a.sqrt()), &expected, 1e-5, "sqrt");
}

#[test]
fn sum_reduce_last_axis() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = a.sum(1).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(b.id), &[6.0, 15.0], 1e-5, "sum_reduce");
}

#[test]
fn max_reduce_last_axis() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = a.max(1).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(b.id), &[5.0, 6.0], 1e-5, "max_reduce");
}

#[test]
fn modulo_binary() {
    let result = run_binary(
        &[10.0, 7.0, 15.0, 3.0, 8.0, 11.0],
        &[3.0, 4.0, 5.0, 2.0, 3.0, 4.0],
        |a, b| a % b,
    );
    assert_close(&result, &[1.0, 3.0, 0.0, 1.0, 2.0, 3.0], 1e-4, "mod");
}

#[test]
fn cast_to_f32() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let out = a.cast(DType::F32).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    rt.set_data_f32(a.id, &data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &data, 1e-4, "cast");
}

#[test]
fn iota_contiguous() {
    let mut cx = Graph::new();
    let out = cx.arange(8).cast(DType::F32).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.execute(&cx.dyn_map);

    let expected: Vec<f32> = (0..8).map(|i| i as f32).collect();
    assert_close(&rt.get_f32(out.id), &expected, 1e-4, "iota");
}

#[test]
fn iota_strided() {
    let head_dim = 8usize;
    let mut cx = Graph::new();
    let out = cx.arange_options(0, head_dim, 2).cast(DType::F32).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &[0.0, 2.0, 4.0, 6.0], 1e-4, "iota step");
}

#[test]
fn gather_with_int_indices() {
    let mut cx = Graph::new();
    let idx = cx.tensor(3).as_dtype(DType::Int);
    let data = cx.tensor(8);
    let out = data.gather(idx).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    let data_vals: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
    rt.set_data_f32(idx.id, &[0.0, 4.0, 6.0]);
    rt.set_data_f32(data.id, &data_vals);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &[10.0, 30.0, 40.0], 1e-4, "gather");
}

#[test]
fn scatter_static_offsets() {
    // Scatter new K rows into the front of a KV cache at flat offsets 0..seq.
    let (n_kv, seq, head_dim, max_seq) = (2usize, 4usize, 4usize, 8usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let k_new = cx.tensor((n_kv, seq, head_dim));
            let k_cache = cx.tensor((n_kv, max_seq, head_dim));
            let h_off = cx.arange(n_kv) * (max_seq * head_dim);
            let p_off = cx.arange(seq) * head_dim;
            let d_off = cx.arange(head_dim);
            let idx = h_off
                .expand_dim(1, seq)
                .expand_dim(2, head_dim)
                + p_off.expand_dim(0, n_kv).expand_dim(2, head_dim)
                + d_off.expand_dim(0, n_kv).expand_dim(1, seq);
            let out = k_new.scatter(idx, k_cache).output();
            (
                vec![
                    (k_new.id, gen_data(n_kv * seq * head_dim, 0.1)),
                    (k_cache.id, gen_data(n_kv * max_seq * head_dim, 0.2)),
                ],
                out.id,
            )
        },
        1e-5,
        "scatter",
    );
}

#[test]
fn matmul_small_2d() {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((3, 2));
    let c = a.matmul(b).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(a.id, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt.set_data_f32(b.id, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(c.id), &[58.0, 64.0, 139.0, 154.0], 1e-4, "matmul");
}

#[test]
fn matmul_batched_3d() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 2, 3));
    let b = cx.tensor((2, 3, 2));
    let c = a.matmul(b).output();

    let mut rt = compile(&mut cx, CompileOptions::default());
    rt.set_data_f32(
        a.id,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    );
    rt.set_data_f32(
        b.id,
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    );
    rt.execute(&cx.dyn_map);

    assert_close(
        &rt.get_f32(c.id),
        &[58.0, 64.0, 139.0, 154.0, 2.0, 2.0, 4.0, 4.0],
        1e-3,
        "batched matmul",
    );
}

#[test]
fn gqa_expand_merge() {
    // KV-head expansion for grouped-query attention: expand then merge heads.
    let (n_kv_heads, kv_groups, seq, head_dim) = (4usize, 2usize, 4usize, 8usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let k = cx.tensor((n_kv_heads, seq, head_dim));
            let out = k.expand_dim(1, kv_groups).merge_dims(0, 1).output();
            (vec![(k.id, gen_data(n_kv_heads * seq * head_dim, 0.2))], out.id)
        },
        1e-5,
        "gqa expand+merge",
    );
}

#[test]
fn transpose_3d() {
    let (n_kv_heads, kv_groups, seq, head_dim) = (4usize, 2usize, 4usize, 8usize);
    assert_matches_reference(
        |cx: &mut Graph| {
            let k = cx.tensor((n_kv_heads, seq, head_dim));
            let out = k
                .expand_dim(1, kv_groups)
                .merge_dims(0, 1)
                .transpose(1, 2)
                .output();
            (vec![(k.id, gen_data(n_kv_heads * seq * head_dim, 0.2))], out.id)
        },
        1e-5,
        "transpose(1,2) on 3D",
    );
}

#[test]
fn gemm_bias_relu_matches_reference() {
    // a.matmul(b) + bias (broadcast over rows) + relu — the egglog rules should
    // fuse all of this into a single MojoGemm kernel (base → bias → relu).
    assert_matches_reference(
        |cx: &mut Graph| {
            let a = cx.tensor((4, 8));
            let b = cx.tensor((8, 3));
            let bias = cx.tensor(3);
            let out = (a.matmul(b) + bias.expand_dim(0, 4)).relu().output();
            (
                vec![
                    (a.id, gen_data(32, 1.0)),
                    (b.id, gen_data(24, 2.0)),
                    // shift negative so some outputs land below zero and relu fires
                    (bias.id, gen_data(3, 3.0).iter().map(|x| x - 0.6).collect()),
                ],
                out.id,
            )
        },
        1e-4,
        "gemm_bias_relu",
    );
}

#[test]
fn gemm_batched_matches_reference() {
    // Batched 3D matmul: out[b,m,n] = a[b,m,k] @ b[b,k,n] — the 3D base rule
    // should fuse this into a single batched MojoGemm kernel.
    assert_matches_reference(
        |cx: &mut Graph| {
            let a = cx.tensor((2, 4, 8));
            let b = cx.tensor((2, 8, 3));
            let out = a.matmul(b).output();
            (
                vec![
                    (a.id, gen_data(64, 1.0)),
                    (b.id, gen_data(48, 2.0)),
                ],
                out.id,
            )
        },
        1e-4,
        "gemm_batched",
    );
}

#[test]
fn rmsnorm_matches_reference() {
    // RMSNorm: out[r,c] = x[r,c] * (mean(x[r,:]²)+eps)^-0.5 — the backend's
    // rewrite should fuse this into a single MojoRMSNorm kernel.
    assert_matches_reference(
        |cx: &mut Graph| {
            let a = cx.tensor((4, 8));
            let ms = ((a * a).mean(1) + 1e-6f32).sqrt().reciprocal();
            let out = (a * ms.expand_dim(1, 8)).output();
            (vec![(a.id, gen_data(32, 1.0))], out.id)
        },
        1e-4,
        "rmsnorm",
    );
}

#[test]
fn softmax_matches_reference() {
    // Softmax: out[r,c] = exp(x[r,c]-max(x[r,:])) / Σ_c exp(x[r,c]-max(x[r,:]))
    // — the backend's rewrite should fuse this into a single MojoSoftmax kernel.
    assert_matches_reference(
        |cx: &mut Graph| {
            let a = cx.tensor((4, 8));
            let out = a.softmax(1).output();
            (vec![(a.id, gen_data(32, 1.0))], out.id)
        },
        1e-4,
        "softmax",
    );
}
