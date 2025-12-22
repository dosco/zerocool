#!/usr/bin/env python3
# scripts/compile_pallas.py
"""
Compile Pallas kernels to StableHLO for PJRT execution.
"""

import jax
from jax.experimental import pallas as pl
import sys
import os
import argparse

def export_kernel_to_hlo(kernel_fn, grid_spec, block_spec, input_shapes, output_shapes, output_path):
    """Export a Pallas kernel to serialized StableHLO."""

    # Create Pallas call with block spec
    # Note: pallas_call API might vary by JAX version.
    # This is a simplified usage based on the design doc.
    
    def pallas_wrapper(*args):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(output_shapes, jax.numpy.float16),
            grid=grid_spec,
            in_specs=[block_spec for _ in args],
            out_specs=block_spec,
        )(*args)

    # JIT and lower to StableHLO
    jitted = jax.jit(pallas_wrapper)
    
    dummy_inputs = [jax.numpy.zeros(s, dtype=jax.numpy.float16) for s in input_shapes]
    
    lowered = jitted.lower(*dummy_inputs)
    hlo_module = lowered.compiler_ir()

    # Serialize
    bytecode = hlo_module.as_serialized_hlo_module_proto()

    with open(output_path, 'wb') as f:
        f.write(bytecode)

    print(f"Exported {kernel_fn.__name__} -> {output_path}")


# TPU-specific block sizes (must align to MXU: 128x128)
TPU_BLOCK_SPEC = pl.BlockSpec((128, 128), lambda i, j: (i, j))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="compiled/stablehlo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Example usage (commented out as we don't have the kernels imported yet)
    # from kernels.pallas.attention import flash_attention_kernel
    # export_kernel_to_hlo(...)
    
    print("Pallas compilation script ready.")
