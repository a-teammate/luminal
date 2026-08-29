//! Shared helpers for luminal_mojo integration tests and benchmarks.

#![allow(dead_code)]

use luminal::hlir::ReferenceRuntime;
use luminal::prelude::*;
use luminal_mojo::MojoRuntime;

/// Deterministic pseudo-data in [0, 1).
pub fn gen_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.01) % 1.0).collect()
}

pub fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compile a graph into a MojoRuntime (canonical single search).
pub fn compile(cx: &mut Graph, opts: CompileOptions) -> MojoRuntime {
    cx.build_search_space::<MojoRuntime>(opts.clone());
    cx.search(MojoRuntime::new(), opts)
}

pub fn assert_close(got: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(got.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (a, b)) in got.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{ctx}[{i}]: got {a}, expected {b} (tol {tol})"
        );
    }
}

/// Build the graph twice with `build`, run one copy on ReferenceRuntime and
/// the other (identically seeded) on MojoRuntime, and return both outputs.
pub fn run_reference_and_mojo(
    build: impl Fn(&mut Graph) -> (Vec<(NodeIndex, Vec<f32>)>, NodeIndex),
) -> (Vec<f32>, Vec<f32>) {
    let mut cx_ref = Graph::new();
    let (inputs, out) = build(&mut cx_ref);
    cx_ref.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt_ref = ReferenceRuntime::default();
    rt_ref = cx_ref.search(rt_ref, CompileOptions::default());
    for (id, data) in &inputs {
        rt_ref.set_data(*id, data.clone());
    }
    rt_ref.execute(&cx_ref.dyn_map);
    let ref_out = rt_ref.get_f32(out).clone();

    let mut cx_mojo = Graph::new();
    let (inputs, out) = build(&mut cx_mojo);
    let mut rt_mojo = compile(&mut cx_mojo, CompileOptions::default());
    for (id, data) in &inputs {
        rt_mojo.set_data_f32(*id, data);
    }
    rt_mojo.execute(&cx_mojo.dyn_map);

    (ref_out, rt_mojo.get_f32(out))
}

/// Assert Mojo output matches the reference output within `tol`.
pub fn assert_matches_reference(
    build: impl Fn(&mut Graph) -> (Vec<(NodeIndex, Vec<f32>)>, NodeIndex),
    tol: f32,
    ctx: &str,
) {
    let (ref_out, mojo_out) = run_reference_and_mojo(build);
    assert_close(&mojo_out, &ref_out, tol, ctx);
}
