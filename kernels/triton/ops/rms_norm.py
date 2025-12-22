import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_w, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    x_ptr = X_ptr + row_idx * stride_x_row
    out_ptr = Out_ptr + row_idx * stride_out_row
    
    # Load data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + offsets * stride_w, mask=mask, other=0.0)
    
    # Compute mean squared
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    
    # Compute scale
    scale = tl.rsqrt(mean_sq + eps)
    
    # Compute output
    y = x * scale * w
    
    tl.store(out_ptr + offsets, y, mask=mask)
