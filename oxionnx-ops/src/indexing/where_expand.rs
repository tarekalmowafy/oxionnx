use oxionnx_core::Tensor;

/// Compute strides for broadcasting a (possibly lower-rank) `shape` up to
/// `ndim` dimensions: left-pad with implicit size-1 axes, and use stride 0
/// on any axis where the operand's size is 1 (broadcast), so walking the
/// *output's* coordinates and mapping them through these strides naturally
/// repeats that axis's single slice across the output — real NumPy/ONNX
/// broadcasting, not flat-index modulo (which is only correct when an
/// operand is a scalar or already full-size).
fn padded_broadcast_strides(shape: &[usize], ndim: usize) -> Vec<usize> {
    let pad = ndim - shape.len();
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..shape.len()).rev() {
        strides[pad + i] = if shape[i] == 1 { 0 } else { s };
        s *= shape[i];
    }
    strides
}

/// Where: select elements from `x` or `y` based on `condition` (bool tensor),
/// with full N-D broadcasting (NumPy/ONNX rules) across all three operands.
pub fn where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(
        &Tensor::broadcast_shape(&condition.shape, &x.shape)?,
        &y.shape,
    )?;
    let ndim = out_shape.len();
    let n: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; n];
    if n == 0 {
        return Ok(Tensor::new(data, out_shape));
    }
    let strides = [
        padded_broadcast_strides(&condition.shape, ndim),
        padded_broadcast_strides(&x.shape, ndim),
        padded_broadcast_strides(&y.shape, ndim),
    ];
    // Prototype (onnx_spike): resolve the operand offsets once per output row
    // (all axes but the last), then walk the row with fixed strides, in parallel.
    let row_len = out_shape.last().copied().unwrap_or(1);
    let inner = [0, 1, 2].map(|o| strides[o].last().copied().unwrap_or(0));
    let (c, xd, yd) = (&condition.data[..], &x.data[..], &y.data[..]);
    crate::par::for_each_row(&mut data, row_len, |row_idx, row| {
        let mut base = [0usize; 3];
        let mut rem = row_idx;
        for d in (0..ndim.saturating_sub(1)).rev() {
            let coord = rem % out_shape[d];
            rem /= out_shape[d];
            for (b, st) in base.iter_mut().zip(&strides) {
                *b += coord * st[d];
            }
        }
        for (j, v) in row.iter_mut().enumerate() {
            *v = if c[base[0] + j * inner[0]] != 0.0 { xd[base[1] + j * inner[1]] } else { yd[base[2] + j * inner[2]] };
        }
    });

    Ok(Tensor::new(data, out_shape))
}

/// Expand: broadcast x to shape.
pub fn expand(x: &Tensor, shape: &[usize]) -> Result<Tensor, String> {
    let out_shape = Tensor::broadcast_shape(&x.shape, shape)?;
    let ndim = out_shape.len();
    let n: usize = out_shape.iter().product();

    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = s;
        s *= out_shape[i];
    }
    let in_strides = padded_broadcast_strides(&x.shape, ndim);

    let mut out = vec![0.0f32; n];
    for (out_idx, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += coord * in_strides[d];
        }
        *out_val = x.data[in_idx];
    }
    Ok(Tensor::new(out, out_shape))
}
