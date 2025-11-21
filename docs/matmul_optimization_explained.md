# Matrix Multiplication Optimization Explained

This document explains the low-level optimizations implemented in `freellm` to speed up matrix multiplication (`matmul`). We achieved a **~12x speedup** (from ~1.8 GFLOPS to ~21 GFLOPS) on ARM NEON by using techniques called **SIMD**, **Loop Unrolling**, and **Register Blocking**.

## 1. The Problem: Memory is Slow

The naive way to multiply two matrices $A$ ($M \times K$) and $B$ ($K \times N$) looks like this:

```cpp
// Naive Triple Loop
for (int i = 0; i < M; i++) {             // For each row of A
    for (int j = 0; j < N; j++) {         // For each column of B
        float sum = 0;
        for (int k = 0; k < K; k++) {     // Dot product
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```

### Why is this slow?
1.  **One operation at a time**: The CPU processes one multiplication and one addition at a time.
2.  **Memory Traffic**: For every single multiplication, we fetch data from memory. Memory (RAM) is hundreds of times slower than the CPU. Even L1 cache is slower than CPU registers.
3.  **Loop Overhead**: The CPU spends a lot of time checking `k < K` and incrementing `k`.

---

## 2. Optimization 1: SIMD (Single Instruction, Multiple Data)

Modern CPUs have special vector registers that can hold multiple numbers at once.
*   **AVX2 (x86)**: 256-bit registers (holds 8 floats).
*   **NEON (ARM)**: 128-bit registers (holds 4 floats).

Instead of doing `a * b`, we can do `[a1, a2, a3, a4] * [b1, b2, b3, b4]` in a **single CPU cycle**.

**Impact**: Theoretically 4x (NEON) or 8x (AVX2) faster math.

---

## 3. Optimization 2: Loop Unrolling

Instead of checking the loop condition every single time, we can repeat the code inside the loop.

**Naive Loop:**
```cpp
for (int k = 0; k < 100; k++) {
    sum += A[k] * B[k]; // Check 'k < 100' 100 times
}
```

**Unrolled Loop:**
```cpp
for (int k = 0; k < 100; k += 4) {
    sum += A[k] * B[k];
    sum += A[k+1] * B[k+1];
    sum += A[k+2] * B[k+2];
    sum += A[k+3] * B[k+3];
    // Check 'k < 100' only 25 times!
}
```

**Impact**: Less time spent on "administrative" work (loop checks), more time on math.

---

## 4. Optimization 3: Register Blocking (The Big One)

This is where the biggest gains come from. It addresses the **Memory is Slow** problem.

CPU Registers are the fastest storage available (instant access). We want to load data into registers once and reuse it as much as possible before throwing it away.

### The "Naive" Memory Pattern
In the standard inner loop, to compute **1** value of C, we load **K** values from A and **K** values from B.
*   **Ratio**: 2 Loads per 1 Multiply-Add.

### The "Register Blocked" Pattern (4x4)
We compute a **4x4 block** of C simultaneously.
*   We load **4** values from A (into registers).
*   We load **4** values from B (into registers).
*   We can perform **16** multiplications (4 rows $\times$ 4 columns) using just those loaded values!

**Analogy**:
*   **Naive**: Go to the grocery store, buy 1 egg, come home. Go back, buy 1 egg, come home.
*   **Blocked**: Go to the store, buy a dozen eggs, come home. You saved 11 trips.

### How it looks in our code (Simplified)

We process **4 rows of A** and **4 columns of B** at the same time.

```cpp
// We use 16 registers to hold a 4x4 block of C (accumulators)
// C_00, C_01... C_33 are all in registers!

for (int k = 0; k < K; k++) {
    // 1. Load 4 values from B (one column strip) into a register
    vec_b = load(B[k]); 
    
    // 2. Load 4 scalars from A (one from each row)
    val_a0 = A[row0][k];
    val_a1 = A[row1][k];
    val_a2 = A[row2][k];
    val_a3 = A[row3][k];

    // 3. Update 4 rows of C at once!
    // Notice we reused 'vec_b' 4 times!
    C_row0 += val_a0 * vec_b;
    C_row1 += val_a1 * vec_b;
    C_row2 += val_a2 * vec_b;
    C_row3 += val_a3 * vec_b;
}
```

**Impact**: 
*   We drastically reduce the number of times we have to fetch `B` from memory.
*   We keep the CPU's arithmetic units busy 100% of the time.

## Summary of Results

| Optimization | What it does | Speedup Factor |
| :--- | :--- | :--- |
| **Naive** | One op at a time | 1x (Baseline) |
| **SIMD** | 4 ops at a time | ~3-4x |
| **Register Blocking** | Reuses data in registers | **~10-12x** (Combined) |

This is why our optimized NEON implementation reached **~21 GFLOPS** compared to the naive **~1.8 GFLOPS**.
