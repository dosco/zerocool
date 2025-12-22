import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    X_ptr, Out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    y = x / (1.0 + tl.exp(-x))
    
    tl.store(Out_ptr + offsets, y, mask=mask)
