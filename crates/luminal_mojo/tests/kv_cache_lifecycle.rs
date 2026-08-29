//! KV-cache lifecycle and HLIR-keyed runtime API tests.
//!
//! Exercises the data API the QwenRuntime trait is built on:
//!   - `set_data_f32`  : F32 weight / activation input
//!   - `set_i32_data`  : token-ID / position-ID input (i32 → f32)
//!   - `set_zeros`     : KV-cache zero-initialisation
//!   - `remove_buffer` / `set_buffer` : KV-cache promote (output → next-step input)
//!   - `get_f32`       : output read-back
//!   - `prepare_execute` : pre-exec hook
//!
//! Weights are seeded programmatically (no model file needed except in the
//! safetensors test). The lifecycle tests drive prefill → KV-cache promote →
//! decode through the multi-bucket (DimBuckets) code path the real example
//! uses.

mod common;

use common::{assert_close, compile};
use luminal::prelude::*;

/// Scatter new K rows into a [n_kv, max_seq, head_dim] cache at flat positions
/// `[prev .. prev+s]`. Verifies the scatter index — which depends on the
/// symbolic position dim 'p' — resolves correctly under bucket switching.
#[test]
fn kv_scatter_with_dynamic_position_across_buckets() {
    let n_kv = 2usize;
    let max_seq = 8usize;
    let head_dim = 2usize;

    let mut cx = Graph::new();
    let k_new = cx.named_tensor("k_new", (n_kv, 's', head_dim));
    let k_cache_in = cx.tensor((n_kv, max_seq, head_dim));

    let s = k_new.dims()[1];
    let prev = Expression::from('p');

    let h_off = cx.arange(n_kv) * (max_seq * head_dim);
    let p_off = (cx.arange(s) + prev) * head_dim;
    let d_off = cx.arange(head_dim);
    let scatter_idx = h_off.expand_dim(1, s).expand_dim(2, head_dim)
        + p_off.expand_dim(0, n_kv).expand_dim(2, head_dim)
        + d_off.expand_dim(0, n_kv).expand_dim(1, s);
    let out = k_new.scatter(scatter_idx, k_cache_in).output();

    let opts = CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(2, 2), DimBucket::new(1, 1)])
        .dim_buckets('p', &[DimBucket::new(0, 0), DimBucket::new(2, 2)]);

    let mut rt = compile(&mut cx, opts);

    rt.set_data_f32(k_cache_in.id, &vec![0.0; n_kv * max_seq * head_dim]);

    // Prefill: s=2, p=0 → write tokens 0,1 into both heads.
    cx.set_dim('s', 2);
    cx.set_dim('p', 0);
    rt.set_data_f32(
        k_new.id,
        &[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
    );
    rt.prepare_execute(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let a = rt.get_f32(out.id);

    let head1 = max_seq * head_dim; // 16
    assert!((a[0] - 10.0).abs() < 1e-3, "head0[0]={}", a[0]);
    assert!((a[3] - 13.0).abs() < 1e-3, "head0[3]={}", a[3]);
    assert!((a[head1] - 20.0).abs() < 1e-3, "head1[0]={}", a[head1]);
    assert!((a[head1 + 3] - 23.0).abs() < 1e-3, "head1[3]={}", a[head1 + 3]);
    assert!(a[4].abs() < 1e-3, "head0 pos2 should be 0, got {}", a[4]);
    assert!(
        a[head1 + 4].abs() < 1e-3,
        "head1 pos2 should be 0, got {}",
        a[head1 + 4]
    );

    // Promote the cache: pull the scatter output and feed it back as the input.
    let buf = rt.remove_buffer(out.id);
    rt.set_buffer(k_cache_in.id, buf);

    // Decode: s=1, p=2 → write token 2, positions 0,1 must survive.
    cx.set_dim('s', 1);
    cx.set_dim('p', 2);
    rt.set_data_f32(k_new.id, &[30.0, 31.0, 40.0, 41.0]);
    rt.prepare_execute(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let b = rt.get_f32(out.id);

    assert!((b[0] - 10.0).abs() < 1e-3, "after decode head0[0]={}", b[0]);
    assert!((b[3] - 13.0).abs() < 1e-3, "after decode head0[1b]={}", b[3]);
    assert!((b[4] - 30.0).abs() < 1e-3, "head0 pos2 (new)={}", b[4]);
    assert!((b[5] - 31.0).abs() < 1e-3, "head0 pos2 (new)={}", b[5]);
    assert!((b[head1 + 4] - 40.0).abs() < 1e-3, "head1 pos2 (new)={}", b[head1 + 4]);
    assert!((b[head1 + 5] - 41.0).abs() < 1e-3, "head1 pos2 (new)={}", b[head1 + 5]);
}

/// Single-layer Qwen-style transformer (all F32): run a real prefill
/// (seq=prompt) then a decode (seq=1) step, promoting the KV cache between
/// them via `remove_buffer`/`set_buffer`.
#[test]
fn single_layer_prefill_then_decode_with_kv_promote() {
    let dim = 16;
    let n_heads = 4;
    let head_dim = 4;
    let n_kv_heads = 2;
    let kv_groups = n_heads / n_kv_heads;
    let intermediate = 32;
    let vocab = 16;
    let max_seq = 16;
    let prompt_len = 4usize;

    let mut cx = Graph::new();
    let seq = Expression::from('s');
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    // ── Inputs ──────────────────────────────────────────────────────────────
    let embedding = cx.tensor((vocab, dim));
    let token_ids = cx.named_tensor("token_ids", 's'); // seeded via set_i32_data

    let h_off = (token_ids.cast(DType::Int) * dim).expand_dim(1, dim);
    let d_off = cx.arange(dim).expand_dim(0, seq);
    let mut x = embedding.gather(h_off + d_off); // [s, dim]

    // ── One transformer block ───────────────────────────────────────────────
    let ms = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
    let attn_w = cx.tensor((dim,));
    let normed = x * ms.expand_dim(1, dim) * attn_w.expand_dim(0, seq);

    let q_proj = cx.tensor((n_heads * head_dim, dim));
    let k_proj = cx.tensor((n_kv_heads * head_dim, dim));
    let v_proj = cx.tensor((n_kv_heads * head_dim, dim));
    let q = normed.matmul(q_proj.t());
    let k = normed.matmul(k_proj.t());
    let v = normed.matmul(v_proj.t());
    let q_3d = q.split_dims(1, head_dim).transpose(0, 1); // [nh, s, hd]
    let k_3d = k.split_dims(1, head_dim).transpose(0, 1); // [nkh, s, hd]
    let v_3d = v.split_dims(1, head_dim).transpose(0, 1);

    // KV cache: [n_kv, max_seq, head_dim]
    let k_cache_in = cx.tensor((n_kv_heads, max_seq, head_dim));
    let v_cache_in = cx.tensor((n_kv_heads, max_seq, head_dim));
    let hc = cx.arange(n_kv_heads) * (max_seq * head_dim);
    let pc = (cx.arange(seq) + prev) * head_dim;
    let dc = cx.arange(head_dim);
    let scatter_idx = hc.expand_dim(1, seq).expand_dim(2, head_dim)
        + pc.expand_dim(0, n_kv_heads).expand_dim(2, head_dim)
        + dc.expand_dim(0, n_kv_heads).expand_dim(1, seq);
    let k_cache_out = k_3d.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_3d.scatter(scatter_idx, v_cache_in);

    let k_full = k_cache_out.slice((.., ..total_seq, ..));
    let v_full = v_cache_out.slice((.., ..total_seq, ..));
    let k_exp = k_full.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
    let v_exp = v_full.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;

    let scale = (head_dim as f32).sqrt().recip();
    let scores = q_3d.matmul(k_exp.transpose(1, 2)) * scale; // [nh, s, total]

    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let attn_dim = k_full.dims()[1];
    let k_pos = cx.arange(attn_dim).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, attn_dim));
    let masked = scores + mask.cast(DType::F32).expand_dim(0, n_heads) * (-1e10f32);
    let attn_out = masked.softmax(2).matmul(v_exp);
    let attn_flat = attn_out.transpose(0, 1).merge_dims(1, 2); // [s, nh*hd]

    let o_proj = cx.tensor((dim, n_heads * head_dim));
    x = x + attn_flat.matmul(o_proj.t());

    // SwiGLU MLP
    let ms2 = ((x * x).mean(1) + 1e-6f32).sqrt().reciprocal();
    let mlp_w = cx.tensor((dim,));
    let mlp_normed = x * ms2.expand_dim(1, dim) * mlp_w.expand_dim(0, seq);
    let gate_w = cx.tensor((intermediate, dim));
    let up_w = cx.tensor((intermediate, dim));
    let down_w = cx.tensor((dim, intermediate));
    let act = mlp_normed.matmul(gate_w.t()).swish() * mlp_normed.matmul(up_w.t());
    x = x + act.matmul(down_w.t());

    // Tied LM head
    let logits = x.matmul(embedding.t()); // [s, vocab]
    let logits_out = logits.output();
    let k_co = k_cache_out.output();
    let v_co = v_cache_out.output();

    // ── Compile: point buckets so representative == actual dim (correctness) ─
    let opts = CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(prompt_len, prompt_len), DimBucket::new(1, 1)])
        .dim_buckets('p', &[DimBucket::new(0, 0), DimBucket::new(prompt_len, prompt_len)]);
    let mut rt = compile(&mut cx, opts);

    // ── Seed weights & zero-init KV cache ───────────────────────────────────
    rt.set_zeros(k_cache_in.id, n_kv_heads * max_seq * head_dim * 4);
    rt.set_zeros(v_cache_in.id, n_kv_heads * max_seq * head_dim * 4);
    rt.set_data_f32(embedding.id, &(0..vocab * dim).map(|i| (i as f32) * 0.001).collect::<Vec<_>>());
    rt.set_data_f32(attn_w.id, &vec![1.0; dim]);
    rt.set_data_f32(o_proj.id, &(0..dim * n_heads * head_dim).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
    rt.set_data_f32(mlp_w.id, &vec![1.0; dim]);
    rt.set_data_f32(gate_w.id, &(0..intermediate * dim).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
    rt.set_data_f32(up_w.id, &(0..intermediate * dim).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
    rt.set_data_f32(down_w.id, &(0..dim * intermediate).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
    for (id, n) in [
        (q_proj.id, n_heads * head_dim * dim),
        (k_proj.id, n_kv_heads * head_dim * dim),
        (v_proj.id, n_kv_heads * head_dim * dim),
    ] {
        rt.set_data_f32(id, &(0..n).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
    }

    // ── Prefill (s=prompt_len, p=0) ─────────────────────────────────────────
    cx.set_dim('s', prompt_len);
    cx.set_dim('p', 0);
    rt.set_i32_data(token_ids.id, vec![1, 5, 3, 2]);
    rt.prepare_execute(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let prefill_logits = rt.get_f32(logits_out.id);
    assert_eq!(prefill_logits.len(), prompt_len * vocab, "prefill logits length");
    assert_logits_finite(&prefill_logits);

    // KV-cache promote: output → next-step input.
    let k_buf = rt.remove_buffer(k_co.id);
    let v_buf = rt.remove_buffer(v_co.id);
    rt.set_buffer(k_cache_in.id, k_buf);
    rt.set_buffer(v_cache_in.id, v_buf);

    // ── Decode (s=1, p=prompt_len) ──────────────────────────────────────────
    cx.set_dim('s', 1);
    cx.set_dim('p', prompt_len);
    rt.set_i32_data(token_ids.id, vec![7]);
    rt.prepare_execute(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let decode_logits = rt.get_f32(logits_out.id);
    assert_eq!(decode_logits.len(), vocab, "decode logits length");
    assert_logits_finite(&decode_logits);
    assert!(
        decode_logits.iter().any(|v| v.abs() > 1e-6),
        "decode logits all zero"
    );
}

fn assert_logits_finite(logits: &[f32]) {
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] not finite: {v}");
    }
    assert!(
        logits.iter().any(|v| v.abs() > 1e-6),
        "logits all zero (no signal)"
    );
}

/// A dynamic-seq matmul graph compiled into two buckets (seq=1 and seq=16).
/// Executes seq=1, then seq=16, then seq=1 again, checking the runtime
/// switches buckets correctly and reproduces the right answer each time.
#[test]
fn dynamic_seq_bucket_switching() {
    let dim = 4;
    let out_dim = 4;

    let mut cx = Graph::new();
    let x = cx.named_tensor("input", ('s', dim));
    let w = cx.tensor((dim, out_dim));
    let out = x.matmul(w).output(); // [s, out_dim]

    let opts = CompileOptions::default()
        .dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(16, 16)]);
    let mut rt = compile(&mut cx, opts);

    // w = ones → out[i,:] = sum of input row
    let w_data = vec![1.0; dim * out_dim];
    let row = vec![1.0_f32, 2.0, 3.0, 4.0]; // sum = 10
    let expected: Vec<f32> = std::iter::repeat(10.0).take(out_dim).collect();

    fn run(
        rt: &mut luminal_mojo::MojoRuntime,
        cx: &mut Graph,
        x_id: NodeIndex,
        w_id: NodeIndex,
        out_id: NodeIndex,
        x_data: &[f32],
        w_data: &[f32],
        s: usize,
        expected: &[f32],
    ) -> Vec<f32> {
        cx.set_dim('s', s);
        rt.set_data_f32(x_id, x_data);
        rt.set_data_f32(w_id, w_data);
        rt.prepare_execute(&cx.dyn_map);
        rt.execute(&cx.dyn_map);
        let r = rt.get_f32(out_id);
        assert_eq!(r.len(), s * expected.len(), "s={s} length");
        for (i, &v) in r.iter().enumerate() {
            assert!((v - expected[i % expected.len()]).abs() < 1e-3, "s={s}[{i}]={v}");
        }
        r
    }

    // seq=1 → bucket 0
    run(&mut rt, &mut cx, x.id, w.id, out.id, &row, &w_data, 1, &expected);

    // seq=16 → bucket 1
    let row16: Vec<f32> = row.repeat(16);
    run(&mut rt, &mut cx, x.id, w.id, out.id, &row16, &w_data, 16, &expected);

    // seq=1 again → back to bucket 0
    run(&mut rt, &mut cx, x.id, w.id, out.id, &row, &w_data, 1, &expected);
}

/// Create a safetensors file with BF16 data, load it via `load_safetensors`,
/// and verify the BF16→F32 conversion is correct.
#[test]
fn load_safetensors_bf16_weights() {
    use std::io::Write;

    let values_f32 = [1.5f32, -2.25, 0.0, 100.0];
    let values_bf16: Vec<half::bf16> = values_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let raw_bytes: Vec<u8> = values_bf16
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();

    let metadata_str = format!(
        r#"{{"test_weight":{{"dtype":"BF16","shape":[4],"data_offsets":[0,{}]}}}}"#,
        raw_bytes.len()
    );
    let header_len = metadata_str.len() as u64;

    let tmp = std::env::temp_dir().join(format!(
        "test_bf16_{}.safetensors",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(metadata_str.as_bytes()).unwrap();
        f.write_all(&raw_bytes).unwrap();
        f.sync_all().unwrap();
    }

    let mut cx = Graph::new();
    let weight = cx.named_tensor("test_weight", (4,)).as_dtype(DType::Bf16);
    let doubled = weight * 2.0f32;
    doubled.output();

    let mut rt = compile(&mut cx, CompileOptions::default());

    rt.load_safetensors(&cx, tmp.to_str().unwrap());
    std::fs::remove_file(&tmp).ok();

    rt.execute(&FxHashMap::default());

    let result = rt.get_f32(doubled.id);
    let expected: Vec<f32> = values_f32.iter().map(|&v| v * 2.0).collect();
    assert_close(&result, &expected, 1e-2, "bf16 safetensors");
}
