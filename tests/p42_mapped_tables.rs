//! P4.2: large tables read only by `Gather` / `GatherBlockQuantized` stay in
//! the mapped model file (inline `raw_data` or external data) instead of being
//! decoded at load time. Every case compares a file load (mapped) against
//! `load_from_bytes` on the same model with the table inline (always decoded)
//! and against hand-computed rows: the results must be bit-identical.

use oxionnx::{Session, SessionBuilder, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── minimal ONNX protobuf encoder ───────────────────────────────────────────

fn varint(mut v: u64) -> Vec<u8> {
    let mut b = Vec::new();
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            b.push(byte);
            return b;
        }
        b.push(byte | 0x80);
    }
}

fn int_field(field: u32, v: u64) -> Vec<u8> {
    let mut b = varint(u64::from(field << 3));
    b.extend(varint(v));
    b
}

fn bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let mut b = varint(u64::from((field << 3) | 2));
    b.extend(varint(data.len() as u64));
    b.extend(data);
    b
}

/// Where an initializer's raw bytes go.
#[derive(Clone, Copy)]
enum Store<'a> {
    Inline,
    /// External file `name` at byte offset `offset` (the file is written by
    /// the test).
    External(&'a str, usize),
}

fn tensor(name: &str, dtype: u64, shape: &[usize], raw: &[u8], store: Store<'_>) -> Vec<u8> {
    let dims: Vec<u8> = shape.iter().flat_map(|&d| varint(d as u64)).collect();
    let mut t = bytes_field(1, &dims);
    t.extend(int_field(2, dtype));
    t.extend(bytes_field(8, name.as_bytes()));
    match store {
        Store::Inline => t.extend(bytes_field(9, raw)),
        Store::External(file, offset) => {
            for (k, v) in [("location", file.to_string()), ("offset", offset.to_string()), ("length", raw.len().to_string())] {
                let mut e = bytes_field(1, k.as_bytes());
                e.extend(bytes_field(2, v.as_bytes()));
                t.extend(bytes_field(13, &e));
            }
            t.extend(int_field(14, 1)); // data_location = EXTERNAL
        }
    }
    t
}

fn value_info(name: &str, shape: &[usize]) -> Vec<u8> {
    let mut tt = int_field(1, 1);
    let dims: Vec<u8> = shape.iter().flat_map(|&d| bytes_field(1, &int_field(1, d as u64))).collect();
    tt.extend(bytes_field(2, &dims));
    let mut vi = bytes_field(1, name.as_bytes());
    vi.extend(bytes_field(2, &bytes_field(1, &tt)));
    vi
}

fn int_attr(name: &str, v: i64) -> Vec<u8> {
    let mut a = bytes_field(1, name.as_bytes());
    a.extend(int_field(3, v as u64));
    a.extend(int_field(20, 2)); // type = INT
    a
}

fn node(op: &str, domain: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        n.extend(bytes_field(1, i.as_bytes()));
    }
    for o in outputs {
        n.extend(bytes_field(2, o.as_bytes()));
    }
    n.extend(bytes_field(3, format!("{op}_{}", outputs[0]).as_bytes()));
    n.extend(bytes_field(4, op.as_bytes()));
    for a in attrs {
        n.extend(bytes_field(5, a));
    }
    if !domain.is_empty() {
        n.extend(bytes_field(7, domain.as_bytes()));
    }
    n
}

fn model(nodes: &[Vec<u8>], inits: &[Vec<u8>], inputs: &[(&str, &[usize])], outputs: &[&str]) -> Vec<u8> {
    let mut g = bytes_field(2, b"p42");
    for n in nodes {
        g.extend(bytes_field(1, n));
    }
    for t in inits {
        g.extend(bytes_field(5, t));
    }
    for (name, shape) in inputs {
        g.extend(bytes_field(11, &value_info(name, shape)));
    }
    for o in outputs {
        g.extend(bytes_field(12, &value_info(o, &[])));
    }
    let mut m = int_field(1, 8);
    m.extend(bytes_field(8, &int_field(2, 17)));
    let mut ms = bytes_field(1, b"com.microsoft");
    ms.extend(int_field(2, 1));
    m.extend(bytes_field(8, &ms));
    m.extend(bytes_field(7, &g));
    m
}

// ── fixtures ────────────────────────────────────────────────────────────────

/// 4096 x 257 FP32 table (4.2 MB, over the 4 MiB mapping threshold; odd row
/// length so rows straddle pages and words).
const V: usize = 4096;
const D: usize = 257;

fn table() -> Vec<f32> {
    (0..V * D).map(|i| ((i * 2654435761usize) % 1_000_003) as f32 * 1e-3 - 500.0).collect()
}

fn le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("oxionnx_p42_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

/// Token ids, including a negative one (counts from the end) and repeats.
fn ids() -> Tensor {
    Tensor::new(vec![0.0, 4095.0, -1.0, 17.0, 17.0, 2048.0], vec![2, 3])
}

fn run(s: &Session, inputs: Vec<(&str, Tensor)>, out: &str) -> Tensor {
    let inputs: HashMap<&str, Tensor> = inputs.into_iter().collect();
    s.run(&inputs).expect("run").remove(out).expect("output")
}

fn expected_rows(t: &[f32], ids: &[usize]) -> Vec<f32> {
    ids.iter().flat_map(|&r| t[r * D..(r + 1) * D].iter().copied()).collect()
}

/// `y = Gather(emb, ids)`, the table stored as `store`; returns the `.onnx`
/// path (written to `d`, with the external file next to it).
fn gather_model(d: &Path, store: Store<'_>) -> (PathBuf, Vec<u8>) {
    let t = table();
    let raw = le(&t);
    if let Store::External(file, offset) = store {
        let mut bytes = vec![0xEEu8; offset];
        bytes.extend(&raw);
        std::fs::write(d.join(file), bytes).expect("write data");
    }
    let m = model(
        &[node("Gather", "", &["emb", "ids"], &["y"], &[])],
        &[tensor("emb", 1, &[V, D], &raw, store)],
        &[("ids", &[2, 3])],
        &["y"],
    );
    let inline = model(
        &[node("Gather", "", &["emb", "ids"], &["y"], &[])],
        &[tensor("emb", 1, &[V, D], &raw, Store::Inline)],
        &[("ids", &[2, 3])],
        &["y"],
    );
    let path = d.join("model.onnx");
    std::fs::write(&path, m).expect("write model");
    (path, inline)
}

fn check_gather(store: Store<'_>, tag: &str) {
    let d = dir(tag);
    let (path, inline) = gather_model(&d, store);
    let mapped = SessionBuilder::new().load(&path).expect("load");
    assert!(mapped.weights()["emb"].data.is_empty(), "{tag}: table should stay in the file");
    assert_eq!(mapped.weights()["emb"].shape, vec![V, D]);
    let decoded = SessionBuilder::new().load_from_bytes(&inline).expect("load bytes");
    assert_eq!(decoded.weights()["emb"].data.len(), V * D);
    let want = expected_rows(&table(), &[0, 4095, 4095, 17, 17, 2048]);
    for s in [&mapped, &decoded] {
        for _ in 0..2 {
            let y = run(s, vec![("ids", ids())], "y");
            assert_eq!(y.shape, vec![2, 3, D], "{tag}");
            assert_eq!(y.data, want, "{tag}");
        }
    }
    // Out-of-range ids are an error, as on the decoded table.
    let bad = Tensor::new(vec![0.0, 4096.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]);
    let inputs: HashMap<&str, Tensor> = [("ids", bad)].into_iter().collect();
    assert!(mapped.run(&inputs).is_err());
    // The session can be saved: the table is rebuilt from the file.
    let saved = mapped.to_optimized_bytes().expect("save");
    let reloaded = SessionBuilder::new().load_optimized_from_bytes(&saved).expect("reload");
    assert_eq!(run(&reloaded, vec![("ids", ids())], "y").data, want);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn inline_table_is_mapped() {
    check_gather(Store::Inline, "inline");
}

#[test]
fn external_table_is_mapped_aligned_and_misaligned() {
    check_gather(Store::External("data.bin", 4096), "ext_aligned");
    check_gather(Store::External("data.bin", 3), "ext_misaligned");
}

#[test]
fn keep_weights_decodes() {
    let d = dir("keep");
    let (path, _) = gather_model(&d, Store::External("data.bin", 0));
    let kept = SessionBuilder::new().with_keep_weights(true).load(&path).expect("load");
    assert_eq!(kept.weights()["emb"].data, table());
    let _ = std::fs::remove_dir_all(&d);
}

/// A tied table (also read by another op, here the `B` input of a MatMul and
/// an `Add`) is decoded, and every use sees the same values.
#[test]
fn tied_table_is_decoded() {
    let d = dir("tied");
    let t = table();
    let raw = le(&t);
    std::fs::write(d.join("data.bin"), &raw).expect("write data");
    let m = model(
        &[
            node("Gather", "", &["emb", "ids"], &["y"], &[]),
            node("MatMul", "", &["h", "emb"], &["p"], &[]),
            node("Add", "", &["emb", "emb"], &["e2"], &[]),
        ],
        &[tensor("emb", 1, &[V, D], &raw, Store::External("data.bin", 0))],
        &[("ids", &[2, 3]), ("h", &[1, V])],
        &["y", "p", "e2"],
    );
    let path = d.join("model.onnx");
    std::fs::write(&path, m).expect("write model");
    let s = SessionBuilder::new().load(&path).expect("load");
    assert_eq!(s.weights()["emb"].data, t, "a tied table must be decoded");
    let mut h = vec![0f32; V];
    h[5] = 1.0;
    let inputs: HashMap<&str, Tensor> = [("ids", ids()), ("h", Tensor::new(h, vec![1, V]))].into_iter().collect();
    let out = s.run(&inputs).expect("run");
    assert_eq!(out["y"].data, expected_rows(&t, &[0, 4095, 4095, 17, 17, 2048]));
    assert_eq!(out["p"].data, t[5 * D..6 * D].to_vec());
    assert_eq!(out["e2"].data, t.iter().map(|x| x + x).collect::<Vec<_>>());
    let _ = std::fs::remove_dir_all(&d);
}

/// `GatherBlockQuantized` (8-bit, uint8 zero points): the 4 MiB byte table is
/// mapped, the small scales / zero points are decoded.
#[test]
fn block_quantized_table_is_mapped() {
    const R: usize = 8192;
    const C: usize = 512;
    const BLOCK: usize = 64;
    let kb = C / BLOCK;
    let q: Vec<u8> = (0..R * C).map(|i| ((i * 131 + 7) % 251) as u8).collect();
    let scales: Vec<f32> = (0..R * kb).map(|i| 0.01 + (i % 97) as f32 * 1e-3).collect();
    let zp: Vec<u8> = (0..R * kb).map(|i| (100 + i % 50) as u8).collect();
    let d = dir("gbq");
    std::fs::write(d.join("data.bin"), &q).expect("write data");
    let attrs = [int_attr("bits", 8), int_attr("block_size", BLOCK as i64), int_attr("gather_axis", 0), int_attr("quantize_axis", 1)];
    let build = |store| {
        model(
            &[node("GatherBlockQuantized", "com.microsoft", &["q", "ids", "s", "z"], &["y"], &attrs)],
            &[
                tensor("q", 2, &[R, C], &q, store),
                tensor("s", 1, &[R, kb], &le(&scales), Store::Inline),
                tensor("z", 2, &[R, kb], &zp, Store::Inline),
            ],
            &[("ids", &[2, 3])],
            &["y"],
        )
    };
    let path = d.join("model.onnx");
    std::fs::write(&path, build(Store::External("data.bin", 0))).expect("write model");
    let mapped = SessionBuilder::new().load(&path).expect("load");
    assert!(mapped.weights()["q"].data.is_empty(), "byte table should stay in the file");
    let decoded = SessionBuilder::new().load_from_bytes(&build(Store::Inline)).expect("load bytes");
    let ids = Tensor::new(vec![0.0, 8191.0, -1.0, 17.0, 4000.0, 2.0], vec![2, 3]);
    let a = run(&mapped, vec![("ids", ids.clone())], "y");
    let b = run(&decoded, vec![("ids", ids)], "y");
    assert_eq!(a.shape, vec![2, 3, C]);
    assert_eq!(a.data, b.data);
    let r = 4000;
    let want: Vec<f32> = (0..C).map(|c| (f32::from(q[r * C + c]) - f32::from(zp[r * kb + c / BLOCK])) * scales[r * kb + c / BLOCK]).collect();
    assert_eq!(&a.data[4 * C..5 * C], &want[..]);
    let _ = std::fs::remove_dir_all(&d);
}
