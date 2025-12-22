import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    X_ptr, Out_ptr,
    stride_x_row, stride_out_row,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    x_ptr = X_ptr + row_idx * stride_x_row
    out_ptr = Out_ptr + row_idx * stride_out_row
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # 1. Max
    x_max = tl.max(x, axis=0)
    
    # 2. Exp
    x_exp = tl.exp(x - x_max)
    
    # 3. Sum
    x_sum = tl.sum(x_exp, axis=0)
    
    # 4. Div
    y = x_exp / x_sum
    
    tl.store(out_ptr + offsets, y, mask=mask)
