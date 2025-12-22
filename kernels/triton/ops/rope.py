import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_x_row, stride_out_row,
    head_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes one row (token * head)
    row_idx = tl.program_id(0)
    
    x_ptr = X_ptr + row_idx * stride_x_row
    out_ptr = Out_ptr + row_idx * stride_out_row
    
    # We process pairs. BLOCK_SIZE should cover head_dim.
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load cos/sin (assuming they are pre-broadcasted or match shape)
    # For simplicity, assuming cos/sin are [head_dim] or [seq_len, head_dim]
    # Here we assume they are passed correctly pointing to the right place.
    cos = tl.load(Cos_ptr + offsets, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + offsets, mask=mask, other=0.0)
    
    # Rotate
    # x = [x0, x1, x2, x3, ...]
    # rotate(x) = [-x1, x0, -x3, x2, ...]
    
    # We can implement rotation by swapping and negating.
    # Or we can load x_rotated directly if we knew how to shuffle.
    # In Triton, we can use swizzling or just math.
    
    # Let's try to construct x_rotated.
    # Even indices i: -x[i+1]
    # Odd indices i: x[i-1]
    
    is_even = (offsets % 2) == 0
    
    # This is tricky without general gather/shuffle if BLOCK_SIZE is large.
    # But we can load x twice with different offsets?
    # Or just compute indices.
    
    # rotated_offsets: if even i -> i+1, if odd i -> i-1
    rotated_offsets = tl.where(is_even, offsets + 1, offsets - 1)
    x_rotated = tl.load(x_ptr + rotated_offsets, mask=mask, other=0.0)
    
    # Apply sign flip: if even (originally), we want -x[i+1]. So negate.
    # If odd (originally), we want x[i-1]. No negate.
    
    # Wait, rotate([-x1, x0])
    # i=0 (even): want -x1. x_rotated[0] is x[1]. So -x_rotated[0].
    # i=1 (odd): want x0. x_rotated[1] is x[0]. So x_rotated[1].
    
    sign = tl.where(is_even, -1.0, 1.0)
    x_rotated = x_rotated * sign
    
    y = x * cos + x_rotated * sin
    
    tl.store(out_ptr + offsets, y, mask=mask)
