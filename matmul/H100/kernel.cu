// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -lcuda -o matmul kernel.cu
// (H100 only: wgmma/TMA need the sm_90a target; -lcuda for cuTensorMapEncodeTiled)
// becoming a god in B200 kernel writing
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>   // bf16 baseline; link with -lcublas
#include <cuda/barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr int WARP_SIZE = 32;
typedef __nv_bfloat16 bf16;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// =====================================================================================
//  WHAT WAS WRONG WITH THE ORIGINAL kernel1
// =====================================================================================
//  1) DEAD GRID DIMENSION.  The kernel indexes blockIdx.y / blockIdx.x, but main launched
//     a 1-D grid (grid_size = cdiv(m*n, BM*BN)).  blockIdx.y is then always 0, so only the
//     first strip of 128 rows of C was ever written -- and every one of those blocks raced
//     on the same output tiles.  A 2-D grid is what the indexing assumes.
//
//  2) THE WARP TILE WAS NEVER COVERED.  The lane layout was 8x4 (lane_id/4, lane_id%4), so
//     one pass of a warp covers (8*TM) x (4*TN) = 32x16 outputs.  The warp tile was
//     declared WM x WN = 64x64 = 4096 outputs.  32x16 = 512.  Seven eighths of every warp
//     tile was silently skipped: garbage C, and the register tile sized for the wrong
//     shape.  The fix is the piece the original was missing: a warp REPEATS its lane
//     pattern WITER_M x WITER_N times to tile its whole WM x WN region (exactly what
//     CUTLASS calls the warp sub-tile, or "iterations").  Now enforced by static_assert, so
//     an inconsistent template config fails to compile instead of producing wrong numbers.
//
//  3) SHARED MEMORY OVERFLOW.  BM=BN=128, BK=64 => (128*64 + 64*128)*4B = 64 KiB of
//     __shared__.  The static limit is 48 KiB per block, so that launch config could not
//     even build.  BK=16 here => 16.9 KiB.
//
//  4) 8-WAY BANK CONFLICT ON EVERY A FRAGMENT LOAD.  As[row][k] has row stride BK.  With BK
//     a multiple of 32, (row*BK + k) % 32 is the same bank for every row, so all 8 distinct
//     lane_m values hit one bank.  One pad column (BK+1) makes the stride odd and spreads
//     those rows over 8 banks.  (Transposing A into As[k][m] is the real fix -- it also
//     makes the fragment load a single LDS.128 -- but it needs vectorised staging.)
//
//  5) acc was indexed [i][j] against the un-iterated shape.  It is now
//     [WITER_M*TM][WITER_N*TN], the real per-thread output count.
//
//  6) The benchmark timed a kernel fed by uninitialised cudaMalloc memory, which can be
//     denormal or NaN and skew the timing.  Inputs are now zeroed.
// =====================================================================================

// -------------------------------------------------------------------------------------
//  kernel1 -- corrected general warp-tiled SGEMM.  Handles ANY M, N, K (bounds-checked).
//
//  THE FULL MECHANISM, outermost blocking to innermost:
//
//   global C  ->  BLOCK tile (BM x BN, one thread block, staged in shared memory)
//             ->  WARP tile (WM x WN, one warp, held in that warp's registers)
//             ->  WARP SUB-TILE (LANE_M*TM x LANE_N*TN, one pass of the 32 lanes)
//             ->  THREAD tile (TM x TN, the actual FFMA register block)
//
//  Each level exists to reuse the data the level above it already paid for:
//    * block level : one global load of an A row is reused by all BN columns of the block.
//    * warp level  : a lane's A_frag register is reused across all TN columns it owns, and
//                    its B_frag across all TM rows -> TM*TN FFMAs per TM+TN loads.  That
//                    ratio is the whole point; it is what turns a memory-bound kernel into
//                    a compute-bound one.
//
//  K is walked in slabs of BK so the working set fits in shared memory.  Per slab:
//    (a) cooperatively stage A[BM x BK] and B[BK x BN] into shared memory,
//    (b) __syncthreads() so nobody reads a tile still being written,
//    (c) walk the BK dimension one k at a time: pull TM values of A and TN of B into
//        registers, accumulate the TM*TN outer product,
//    (d) __syncthreads() again before overwriting shared memory on the next slab.
// -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN, int LANE_N>
__global__ __launch_bounds__((BM / WM) * (BN / WN) * WARP_SIZE)
void kernel1(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int M, int N, int K) {
  // ---- block / warp / lane decomposition (all compile-time except the ids) ----
  constexpr int WARPS_M = BM / WM;                // warps stacked along M
  constexpr int WARPS_N = BN / WN;                // warps stacked along N
  constexpr int BLOCK_SIZE = WARPS_M * WARPS_N * WARP_SIZE;

  constexpr int LANE_M = WARP_SIZE / LANE_N;      // the 32 lanes as a LANE_M x LANE_N grid
  constexpr int WSUB_M = LANE_M * TM;             // rows one warp pass covers
  constexpr int WSUB_N = LANE_N * TN;             // cols one warp pass covers
  constexpr int WITER_M = WM / WSUB_M;            // passes needed to fill the warp tile
  constexpr int WITER_N = WN / WSUB_N;

  static_assert(BM % WM == 0 && BN % WN == 0, "block tile must divide into warp tiles");
  static_assert(WARP_SIZE % LANE_N == 0, "lane grid must use exactly 32 lanes");
  static_assert(WM % WSUB_M == 0 && WN % WSUB_N == 0, "warp tile must divide into sub-tiles");

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_row = (warp_id / WARPS_N) * WM;  // warp origin inside the block tile
  const int warp_col = (warp_id % WARPS_N) * WN;
  const int lane_m = lane_id / LANE_N;            // lane origin inside a sub-tile
  const int lane_n = lane_id % LANE_N;

  const int block_row = blockIdx.y * BM;          // blockIdx.y is now real (2-D grid)
  const int block_col = blockIdx.x * BN;

  // +1 pad column: breaks the shared-memory bank conflict described in note (4) above.
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN];

  float acc[WITER_M * TM][WITER_N * TN] = {};     // per-thread output tile, zeroed

  for (int k0 = 0; k0 < K; k0 += BK) {
    // (a) stage the slab.  The flat loop strided by BLOCK_SIZE puts consecutive tids on
    //     consecutive addresses, so the global reads coalesce.  Out-of-range entries are
    //     zero-filled, which makes the ragged edge contribute nothing to the dot products
    //     instead of needing a special-cased inner loop.
    for (int i = tid; i < BM * BK; i += BLOCK_SIZE) {
      int r = i / BK, c = i % BK;
      int gr = block_row + r, gc = k0 + c;
      As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
    }
    for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
      int r = i / BN, c = i % BN;
      int gr = k0 + r, gc = block_col + c;
      Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
    }
    __syncthreads();                              // (b) tiles now visible to the whole block

    // (c) rank-1 updates along the staged K slab
    for (int k = 0; k < BK; k++) {
      float a_frag[WITER_M][TM];
      float b_frag[WITER_N][TN];
      for (int im = 0; im < WITER_M; im++)
        for (int i = 0; i < TM; i++)
          a_frag[im][i] = As[warp_row + im * WSUB_M + lane_m * TM + i][k];
      for (int jn = 0; jn < WITER_N; jn++)
        for (int j = 0; j < TN; j++)
          b_frag[jn][j] = Bs[k][warp_col + jn * WSUB_N + lane_n * TN + j];

      // (TM+TN)*WITER loads feed TM*TN*WITER_M*WITER_N FFMAs -- that is the register reuse.
      for (int im = 0; im < WITER_M; im++)
        for (int i = 0; i < TM; i++)
          for (int jn = 0; jn < WITER_N; jn++)
            for (int j = 0; j < TN; j++)
              acc[im * TM + i][jn * TN + j] += a_frag[im][i] * b_frag[jn][j];
    }
    __syncthreads();                              // (d) safe to overwrite shared memory
  }

  // epilogue: registers -> C, same index arithmetic as the fragment loads
  for (int im = 0; im < WITER_M; im++)
    for (int i = 0; i < TM; i++) {
      int row = block_row + warp_row + im * WSUB_M + lane_m * TM + i;
      if (row >= M) continue;
      for (int jn = 0; jn < WITER_N; jn++)
        for (int j = 0; j < TN; j++) {
          int col = block_col + warp_col + jn * WSUB_N + lane_n * TN + j;
          if (col < N) C[(size_t)row * N + col] = acc[im * TM + i][jn * TN + j];
        }
    }
}


////////////////////////////////////////////////////
////        H100 specific kernels
////////////////////////////////////////////////////

// =====================================================================================
//  __launch_bounds__(MAX_THREADS[, MIN_BLOCKS_PER_SM]) -- what it does, since both kernels
//  now carry it:
//
//  ptxas has to pick a register count per thread WITHOUT knowing the launch config.  An
//  SM has 65536 registers; a block of T threads using R registers each needs T*R of them.
//  If ptxas guesses wrong, two things can go wrong at runtime:
//    * too many registers  -> the launch FAILS ("too many resources requested")
//    * too few, playing safe -> spills to local memory that were never necessary
//  __launch_bounds__(T) is the promise "this kernel is never launched with more than T
//  threads per block", so ptxas may use up to min(255, 65536 / T) registers freely.  The
//  optional second argument M says "I want at least M blocks resident per SM", which caps
//  registers at 65536 / (T * M) -- a knob to trade spills for occupancy.  Launching with
//  more than T threads is a hard error, so the bound is also a compile-time contract
//  between the kernel and its launch site (here both derive from the same template ints).
// =====================================================================================

// -------------------------------------------------------------------------------------
//  kernel2 -- first Hopper-only kernel: TMA loads + WGMMA tensor-core MMA.  bf16 in,
//  fp32 accumulate, bf16 out.  M, N, K must be multiples of BM, BN, BK (no edge handling).
//
//  WHY.  kernel1 is FFMA on CUDA cores: 67 TFLOP/s fp32 peak, we sit at ~26.  Hopper's
//  tensor cores do ~990 TFLOP/s dense bf16.  Three hardware features unlock them, and this
//  kernel uses each in its simplest form (one warpgroup, 64x64 tile, no pipelining) so the
//  mechanism stays visible:
//
//   1) WGMMA  (wgmma.mma_async).  An MMA issued by a WARPGROUP = 4 warps = 128 threads.
//      One instruction does D[64xN] += A[64x16] * B[16xN].  A and B are read straight
//      from SHARED MEMORY through 64-bit "matrix descriptors" (address, strides, swizzle
//      mode); D lives in registers spread over the 128 threads (32 floats each for N=64).
//      It is asynchronous: fence -> issue several -> commit_group -> wait_group.
//
//   2) TMA  (cp.async.bulk.tensor).  ONE thread asks the copy engine for a whole
//      [BM x BK] box of a tensor; the hardware does the address math and the smem writes.
//      No per-thread staging loop, no registers burnt on loads, no coalescing to think
//      about.  Box shape and global strides live in a CUtensorMap built on the host by
//      cuTensorMapEncodeTiled (driver API, hence -lcuda).
//
//   3) mbarrier with TRANSACTION COUNT.  How do 128 threads learn the TMA landed?  The TMA
//      is told which barrier to signal; the barrier completes its phase when every thread
//      has arrived AND the expected byte count (arrive_tx) has been delivered.  This
//      replaces the __syncthreads() after kernel1's staging loop.
//
//  128B SWIZZLE glues (1) and (2).  A tile row is BK*2B = 128 bytes.  Stored plainly,
//  wgmma's smem reads would bank-conflict exactly like note (4) of kernel1.  So the tensor
//  map tells TMA to XOR-swizzle 16-byte chunks inside every 1024-byte group, and bit 62
//  of the wgmma descriptor tells the tensor core to expect the same pattern.  Neither
//  side computes an index; two flags agree on a layout.  Descriptor fields (all in
//  16-byte units, i.e. bytes >> 4):
//      bits  0-13  smem start address
//      bits 16-29  leading byte offset   (unused under swizzle, set to 16)
//      bits 32-45  stride byte offset    1024 = 8 rows x 128B, between 8-row core groups
//      bits 62-63  swizzle mode          1 = 128B
//  Advancing K by 16 inside a slab is just start address += 32 bytes: the swizzle is a
//  function of the address bits, so the bumped descriptor still lands on the right data.
//
//  LAYOUT (differs from kernel1!):  A is M x K row-major and B is N x K row-major -- both
//  "K-major", which is what wgmma with TransA = TransB = 0 consumes.  C is M x N
//  COLUMN-major, element (i,j) at j*M + i.  In row-major terms: C = A * B^T.
//
//  PER K SLAB:
//    (a) thread 0 issues TMA(A) + TMA(B) and arrive_tx(bytes) on each barrier;
//        the other 127 threads just arrive,
//    (b) everyone waits on both barriers -> the slab is in smem,
//    (c) fence, BK/16 wgmma's, commit, wait<0>  (wait<0> also guarantees the tensor
//        core is done reading smem before the next TMA overwrites it),
//  and after the last slab the accumulator fragments are written to C.
// -------------------------------------------------------------------------------------
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ---- wgmma plumbing ----------------------------------------------------------------
__device__ static inline uint64_t desc_field(uint64_t bytes) { return (bytes & 0x3FFFF) >> 4; }

__device__ static inline uint64_t smem_desc(const bf16* p) {
  // we use _cta_generic_to_shared(p) because wgmma needs a shared-memory-address not a generic address
  uint64_t desc = desc_field(__cvta_generic_to_shared(p));
  desc |= desc_field(16) << 16;      // leading byte offset
  desc |= desc_field(1024) << 32;    // stride byte offset
  desc |= 1llu << 62;                // 128B swizzle
  return desc;
}

__device__ static inline void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ static inline void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ static inline void wgmma_wait() {
  static_assert(N >= 0 && N <= 7, "wgmma.wait_group takes 0..7");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Per-warpgroup register reallocation (setmaxnreg): the producer only issues TMA so it
// hands most of its register quota to the consumers, who need room for the deep async
// WGMMA pipeline.  RegCount must be a multiple of 8.  All 128 threads of the warpgroup
// must execute the instruction together.
template <uint32_t RegCount>
__device__ static inline void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
template <uint32_t RegCount>
__device__ static inline void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// D[64x64] += A[64x16] * B[16x64], bf16 operands from swizzled smem, fp32 accumulate.
// FRAGMENT LAYOUT of d (needed by the epilogue): warp w owns rows 16w..16w+15.  Within a
// warp, lane/4 picks the row (0..7) and lane%4 picks a column pair.  For 16-column chunk j:
//   d[j][0..1] = (row,   16j + 2*(lane%4) + {0,1})     d[j][4..5] = same row, +8 columns
//   d[j][2..3] = (row+8, same columns)                  d[j][6..7] = row+8, +8 columns
__device__ static inline void wgmma_m64n64k16(float (&d)[4][8], const bf16* a, const bf16* b) {
  uint64_t desc_a = smem_desc(a), desc_b = smem_desc(b);
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32, %33, 1, 1, 1, 0, 0;\n"   // scale-d, scale-a, scale-b, trans-a, trans-b
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
        "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
        "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
        "+f"(d[3][6]), "+f"(d[3][7])
      : "l"(desc_a), "l"(desc_b));
}

// D[64x128] += A[64x16] * B[16x128].  Same fragment layout as the n64 form, only
// wider: j now runs over 8 column chunks (N/16) instead of 4.
__device__ static inline void wgmma_m64n128k16(float (&d)[8][8], const bf16* a, const bf16* b) {
  uint64_t desc_a = smem_desc(a), desc_b = smem_desc(b);
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0 , %1 , %2 , %3 , %4 , %5 , %6 , %7 ,"
      " %8 , %9 , %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63},"
      " %64, %65, 1, 1, 1, 0, 0;\n"   // scale-d, scale-a, scale-b, trans-a, trans-b
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
        "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
        "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
        "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
        "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
        "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
      : "l"(desc_a), "l"(desc_b));
}

// D[64x256] += A[64x16] * B[16x256].  Widest WGMMA shape: the n64/n128 fragment layout
// documented above, extended to 16 column chunks (N/16).  One instruction per k-step is
// exactly what lets a 128x256 block tile fit two warpgroups -- each thread holds
// 16*8 = 128 accumulator floats, which is the per-thread tile limit.
__device__ static inline void wgmma_m64n256k16(float (&d)[16][8], const bf16* a, const bf16* b) {
  uint64_t desc_a = smem_desc(a), desc_b = smem_desc(b);
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0 , %1 , %2 , %3 , %4 , %5 , %6 , %7 ,"
      " %8 , %9 , %10, %11, %12, %13, %14, %15,"
      " %16, %17, %18, %19, %20, %21, %22, %23,"
      " %24, %25, %26, %27, %28, %29, %30, %31,"
      " %32, %33, %34, %35, %36, %37, %38, %39,"
      " %40, %41, %42, %43, %44, %45, %46, %47,"
      " %48, %49, %50, %51, %52, %53, %54, %55,"
      " %56, %57, %58, %59, %60, %61, %62, %63,"
      " %64, %65, %66, %67, %68, %69, %70, %71,"
      " %72, %73, %74, %75, %76, %77, %78, %79,"
      " %80, %81, %82, %83, %84, %85, %86, %87,"
      " %88, %89, %90, %91, %92, %93, %94, %95,"
      " %96, %97, %98, %99, %100, %101, %102, %103,"
      " %104, %105, %106, %107, %108, %109, %110, %111,"
      " %112, %113, %114, %115, %116, %117, %118, %119,"
      " %120, %121, %122, %123, %124, %125, %126, %127},"
      " %128, %129, 1, 1, 1, 0, 0;\n"   // scale-d, scale-a, scale-b, trans-a, trans-b
      "}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
        "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
        "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
        "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
        "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
        "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]), "+f"(d[8][0]), "+f"(d[8][1]),
        "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]),
        "+f"(d[9][6]), "+f"(d[9][7]), "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
        "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]), "+f"(d[11][0]), "+f"(d[11][1]),
        "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]),
        "+f"(d[12][6]), "+f"(d[12][7]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
        "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]), "+f"(d[14][0]), "+f"(d[14][1]),
        "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
        "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]),
        "+f"(d[15][6]), "+f"(d[15][7])
      : "l"(desc_a), "l"(desc_b));
}

// ---- TMA plumbing (host) ------------------------------------------------------------
// Tensor map over a [rows x K] K-major bf16 matrix, delivering [BOX_ROWS x BOX_K] boxes
// into smem with the 128B swizzle.
template <int BOX_ROWS, int BOX_K>
static CUtensorMap make_tensor_map(const bf16* ptr, int rows, int K) {
  CUtensorMap map;
  uint64_t shape[2]  = {(uint64_t)K, (uint64_t)rows};        // innermost dimension first
  uint64_t stride[1] = {sizeof(bf16) * (uint64_t)K};          // byte stride of dim 1
  uint32_t box[2]    = {BOX_K, BOX_ROWS};
  uint32_t elem[2]   = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
                                      shape, stride, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed (%d)\n", r); exit(1); }
  return map;
}

// ---- the kernel ----------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void kernel2(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_K = 16;
  static_assert(BM == 64 && BN == 64, "one warpgroup, one m64n64 instruction shape");
  static_assert(NUM_THREADS == 128, "exactly one warpgroup");
  static_assert(BK * sizeof(bf16) == 128, "tile row must be 128 bytes for the 128B swizzle");

  // 1024-aligned: the swizzle pattern repeats every 1024 bytes and is computed from raw
  // address bits, so a tile base must sit on a pattern boundary.
  __shared__ alignas(1024) bf16 sA[BM * BK];
  __shared__ alignas(1024) bf16 sB[BN * BK];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier barA, barB;

  float d[BN / 16][8] = {};                       // this thread's slice of the 64x64 tile
  static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float), "accumulators cover the tile");

  const int block_n = blockIdx.x % (N / BN);
  const int block_m = blockIdx.x / (N / BN);

  if (threadIdx.x == 0) {
    init(&barA, NUM_THREADS);
    init(&barB, NUM_THREADS);
    cde::fence_proxy_async_shared_cta();          // barrier init visible to the TMA unit
  }
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += BK) {
    // (a) one thread fires the loads and declares the byte counts; the rest just arrive
    barrier::arrival_token tokA, tokB;
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(sA, &mapA, k0, block_m * BM, barA);
      tokA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
      cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &mapB, k0, block_n * BN, barB);
      tokB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
    } else {
      tokA = barA.arrive();
      tokB = barB.arrive();
    }
    // (b) slab is in smem once both barriers flip
    barA.wait(std::move(tokA));
    barB.wait(std::move(tokB));

    // (c) BK/16 async MMAs, each a 32-byte bump of the descriptor start inside the row
    wgmma_fence();
    for (int kk = 0; kk < BK; kk += WGMMA_K) wgmma_m64n64k16(d, &sA[kk], &sB[kk]);
    wgmma_commit();
    wgmma_wait<0>();
  }

  // epilogue: fragments -> column-major C, following the layout documented at wgmma_m64n64k16
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const int row = warp * 16 + lane / 4;
  bf16* tile = C + (size_t)block_n * BN * M + block_m * BM;
  auto put = [&](int r, int c, float v) { tile[(size_t)c * M + r] = __float2bfloat16(v); };
  for (int j = 0; j < BN / 16; j++) {
    int col = 16 * j + 2 * (lane % 4);
    put(row,     col,     d[j][0]);  put(row,     col + 1, d[j][1]);
    put(row + 8, col,     d[j][2]);  put(row + 8, col + 1, d[j][3]);
    put(row,     col + 8, d[j][4]);  put(row,     col + 9, d[j][5]);
    put(row + 8, col + 8, d[j][6]);  put(row + 8, col + 9, d[j][7]);
  }
}

// -------------------------------------------------------------------------------------
// kernel3 -- one warpgroup, with the kernel2 WGMMA broken into smaller matmuls.
//
// A block owns BM x BN and the WGMMA is as wide as the block (WGMMA_N == BN), so the
// decomposition runs along M and K only:
//
//   [BM/WGMMA_M, BK/WGMMA_K] * [BK/WGMMA_K, 1]
//
// One m64n128k16 replaces the two m64n64k16 that used to be issued per (mi, kk): same
// math, half the instructions, and the B descriptor is read once per K step instead of
// once per N subtile.  Accumulators for all M subtiles stay live until the epilogue,
// so the K loop is shared by every micro-matmul rather than reloading A/B.
// -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void kernel3(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_M = 64, WGMMA_N = BN, WGMMA_K = 16;
  static_assert(NUM_THREADS == 128, "kernel3 uses one warpgroup");
  static_assert(BM % WGMMA_M == 0, "BM must be a multiple of the WGMMA M tile");
  static_assert(WGMMA_N == 128, "the WGMMA spans the whole block tile: only n128 is wired up");
  static_assert(BK % WGMMA_K == 0 && BK * sizeof(bf16) == 128,
                "BK must be WGMMA-compatible and one 128B swizzle row");

  constexpr int MTILES = BM / WGMMA_M;
  __shared__ alignas(1024) bf16 sA[BM * BK];
  __shared__ alignas(1024) bf16 sB[BN * BK];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier barA, barB;

  // d[mi] is the 64-float fragment owned by this thread for one 64xBN micro-matmul.
  float d[MTILES][WGMMA_N / 16][8] = {};
  static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float),
                "micro-matmul fragments must cover the block tile");

  const int block_n = blockIdx.x % (N / BN);
  const int block_m = blockIdx.x / (N / BN);

  if (threadIdx.x == 0) {
    init(&barA, NUM_THREADS);
    init(&barB, NUM_THREADS);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += BK) {
    barrier::arrival_token tokA, tokB;
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(sA, &mapA, k0, block_m * BM, barA);
      tokA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
      cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &mapB, k0, block_n * BN, barB);
      tokB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
    } else {
      tokA = barA.arrive();
      tokB = barB.arrive();
    }
    barA.wait(std::move(tokA));
    barB.wait(std::move(tokB));

    wgmma_fence();
    // The decomposition under test: each call is one 64xBNx16 matrix multiply that
    // spans the block's full N, so only M and K are walked.  All calls share this slab.
    for (int mi = 0; mi < MTILES; ++mi)
      for (int kk = 0; kk < BK; kk += WGMMA_K)
        wgmma_m64n128k16(d[mi], &sA[mi * WGMMA_M * BK + kk], &sB[kk]);
    wgmma_commit();
    wgmma_wait<0>();
  }

  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const int row = warp * 16 + lane / 4;        // row within the 64-row WGMMA tile
  for (int mi = 0; mi < MTILES; ++mi) {
    bf16* tile = C + (size_t)block_n * BN * M + block_m * BM + mi * WGMMA_M;
    auto put = [&](int r, int c, float v) { tile[(size_t)c * M + r] = __float2bfloat16(v); };
    for (int j = 0; j < WGMMA_N / 16; ++j) {
      int col = 16 * j + 2 * (lane % 4);
      put(row,     col,     d[mi][j][0]);  put(row,     col + 1, d[mi][j][1]);
      put(row + 8, col,     d[mi][j][2]);  put(row + 8, col + 1, d[mi][j][3]);
      put(row,     col + 8, d[mi][j][4]);  put(row,     col + 9, d[mi][j][5]);
      put(row + 8, col + 8, d[mi][j][6]);  put(row + 8, col + 9, d[mi][j][7]);
    }
  }
}

/// kernel 4 -> using producer(loads) + consumer(tensor core)
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ __launch_bounds__(NUM_THREADS)
void kernel4(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_M = 64, WGMMA_N = BN, WGMMA_K=16;
  static_assert(NUM_THREADS % 128 == 0 && NUM_THREADS >= 256,
                "kernel4 needs 1 producer warpgroup + at least 1 consumer warpgroup");
  static_assert(BM % WGMMA_M == 0, "BM must be a multiple of the WGMMA M tile");
  static_assert(WGMMA_N == 128, "the WGMMA spans the whole block tile: only n128 is wired up");
  static_assert(BK % WGMMA_K == 0 && BK * sizeof(bf16) == 128,
                "BK must be WGMMA-compatible and one 128B swizzle row");

  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  int warpgroup_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;
  const int MTILES = BM / WGMMA_M;
  const int block_n = blockIdx.x % (N / BN);
  const int block_m = blockIdx.x / (N / BN);
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[QSIZE], empty[QSIZE];
  extern __shared__ __align__(1024) bf16 smem[];
  bf16* As = smem;                      // QSIZE stages of BM x BK
  bf16* Bs = smem + QSIZE * BM * BK;    // QSIZE stages of BK x BN
  float d[MTILES][WGMMA_N / 16][8] = {};
  static_assert(sizeof(d) * num_consumers * 128 == BM * BN * sizeof(float),
                "consumer fragments must cover the block tile");
  if (threadIdx.x == 0){
    for (int i = 0;i<QSIZE;i++){
      init(&full[i], num_consumers * 128 + 1);
      init(&empty[i], num_consumers * 128 + 1);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (warpgroup_idx == 0){
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    if (tid ==0){
      int idx = 0;
      for (int bk_it = 0; bk_it < K / BK; ++bk_it, idx = (idx + 1) % QSIZE) {
        auto token = empty[idx].arrive();
        empty[idx].wait(std::move(token));
        cde::cp_async_bulk_tensor_2d_global_to_shared(&As[idx*BK*BM], &mapA, bk_it*BK, block_m * BM, full[idx]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs[idx*BK*BN], &mapB, bk_it*BK, block_n * BN, full[idx]);
        barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full[idx], 1, (BK*BN+BK*BM)*sizeof(bf16));
      }
    }
  } else {
    for (int i=0;i<QSIZE;++i){
      barrier::arrival_token _ = empty[i].arrive();
    }
    for (int k0 = 0, idx = 0; k0 < K; k0 += BK, idx = (idx + 1) % QSIZE) {
      full[idx].wait(full[idx].arrive());
      wgmma_fence();
      #pragma unroll
      for (int mi = 0;mi<MTILES;++mi){
        bf16 *wgmma_sA = As + idx * BK * BM + mi * WGMMA_M * BK;
        #pragma unroll
        for (int kk=0;kk<BK;kk+=WGMMA_K){
          wgmma_m64n128k16(d[mi], &wgmma_sA[kk], &Bs[idx * BK * BN + kk]);
        }
      }
      wgmma_commit();
      wgmma_wait<0>();
      barrier::arrival_token _ = empty[idx].arrive();
    }
    // In a multi-warpgroup kernel the consumer threads live at threadIdx.x >= 128,
    // so the WGMMA fragment layout needs the warp index WITHIN the consumer
    // warpgroup (tid/32 -> 0..3), not the global (threadIdx.x/32 -> 4..7).
    const int lane = threadIdx.x % WARP_SIZE, warp = tid / WARP_SIZE;
    const int row = warp * 16 + lane / 4;      // row within the 64-row WGMMA tile
    for (int mi = 0; mi < MTILES; ++mi) {
      bf16* tile = C + (size_t)block_n * BN * M + block_m * BM + mi * WGMMA_M;
      auto put = [&](int r, int c, float v) { tile[(size_t)c * M + r] = __float2bfloat16(v); };
      for (int j = 0; j < WGMMA_N / 16; ++j) {
        int col = 16 * j + 2 * (lane % 4);
        put(row,     col,     d[mi][j][0]);  put(row,     col + 1, d[mi][j][1]);
        put(row + 8, col,     d[mi][j][2]);  put(row + 8, col + 1, d[mi][j][3]);
        put(row,     col + 8, d[mi][j][4]);  put(row,     col + 9, d[mi][j][5]);
        put(row + 8, col + 8, d[mi][j][6]);  put(row + 8, col + 9, d[mi][j][7]);
      }
    }
  }
}

// kernel no. 5 -- 128x256 output tile, 1 producer + 2 consumer warpgroups.
//
// WHY.  kernel4 covers a 128x128 output tile with one producer + one consumer
// warpgroup, using m64n128k16 WGMMAs.  Pushing the tile to 128x256 doubles the
// arithmetic per block: 128*256*K FMA ops amortise the same TMA bandwidth twice.
// But a single warpgroup cannot hold 128x256 accumulator fragments (each thread
// would need 16*8 = 128 float registers for 256-wide fragments alone -- that is
// the per-thread tile limit).
//
// SOLUTION: two consumer warpgroups.  After a slab of BK rows lands in smem, the
// 128-row block tile is split along M into two 64-row halves; consumer warpgroup
// wg (wg = 1 or 2) computes half (wg-1) using m64n256k16 WGMMAs -- one
// instruction per k-step is the widest Hopper supports, and halves the number of
// WGMMA issues per slab compared to splitting n128 + n128.
//
// BARrier ARITHMETIC with 2 consumers (256 threads) + 1 producer thread:
//   full[i]  expects 257 arrivals (256 consumer + 1 producer arrive_tx with byte count).
//   empty[i] expects 257 arrivals (256 consumer + 1 producer).
//   Initial empty[] arrival prefetches QSIZE slabs (QSIZE drops to 3 from 5 because
//   each stage is 2x wider: 3 * (128*64 + 64*256) bf16 = 144 KiB dynamic smem,
//   comfortably under H100's 227 KiB cap).
//
// REGISTERS: each consumer thread holds d[16][8] = 128 float accumulators (one
// 64x256 half), leaving ~40 registers for the address/token overhead -- enough
// for __launch_bounds__(384) to fit in 170 regs/thread without spills.
// -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ __launch_bounds__(NUM_THREADS)
void kernel5(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_M = 64, WGMMA_N = BN, WGMMA_K = 16;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  static_assert(num_consumers == 2, "kernel5 needs 2 consumer warpgroups");
  static_assert(BM == num_consumers * WGMMA_M, "each consumer computes a 64-row half");
  static_assert(BK % WGMMA_K == 0 && BK * sizeof(bf16) == 128,
                "BK must be WGMMA-compatible and one 128B swizzle row");

  const int warpgroup_idx = threadIdx.x / 128;
  const int tid            = threadIdx.x % 128;
  const int block_n = blockIdx.x % (N / BN);
  const int block_m = blockIdx.x / (N / BN);
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[QSIZE], empty[QSIZE];
  extern __shared__ __align__(1024) bf16 smem[];
  bf16* As = smem;                        // QSIZE stages of BM x BK
  bf16* Bs = smem + QSIZE * BM * BK;      // QSIZE stages of BK x BN

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; i++) {
      init(&full[i],  num_consumers * 128 + 1);   // 256 consumers + 1 producer arrive_tx
      init(&empty[i], num_consumers * 128 + 1);   // 256 consumers + 1 producer
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (warpgroup_idx == 0) {
    // ---- producer warpgroup: TMA loads, ring-buffer of QSIZE stages ----
    // It only issues TMA, so shrink it to 24 regs and free the file for the consumers.
    warpgroup_reg_dealloc<24>();
    if (tid == 0) {
      int idx = 0;
      for (int bk_it = 0; bk_it < K / BK; ++bk_it, idx = (idx + 1) % QSIZE) {
        auto token = empty[idx].arrive();
        empty[idx].wait(std::move(token));
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &As[idx * BK * BM], &mapA, bk_it * BK, block_m * BM, full[idx]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &Bs[idx * BK * BN], &mapB, bk_it * BK, block_n * BN, full[idx]);
        barrier::arrival_token _ =
            cuda::device::barrier_arrive_tx(full[idx], 1,
                                           (size_t)(BK * BN + BK * BM) * sizeof(bf16));
      }
    }
  } else {
    // ---- consumer warpgroups (wg 1, wg 2): each one 64-row half of the 128x256 tile ----
    // Grab the registers the producer released (2 consumers -> 240) so the async WGMMA
    // pipeline stays deep and spill-free.  d MUST be declared here, AFTER setmaxnreg:
    // if it lives at the function top, ptxas sizes its 128 accumulator registers under
    // the uniform __launch_bounds__ budget and never does the 24/240 split.
    warpgroup_reg_alloc<240>();
    const int mi = warpgroup_idx - 1;  // 0 or 1 → offset within As for the 64-row half
    float d[WGMMA_N / 16][8] = {};
    static_assert(sizeof(d) * num_consumers * 128 == BM * BN * sizeof(float),
                  "consumer fragments must cover the block tile");

    // Prefetch: arrive on every empty[] phase so the producer can fill all QSIZE slabs
    // up front before the consumers start consuming.
    for (int i = 0; i < QSIZE; ++i) barrier::arrival_token _ = empty[i].arrive();

    for (int k0 = 0, idx = 0; k0 < K; k0 += BK, idx = (idx + 1) % QSIZE) {
      full[idx].wait(full[idx].arrive());

      wgmma_fence();
      bf16* wgmma_sA = As + idx * BK * BM + mi * WGMMA_M * BK;
      #pragma unroll
      for (int kk = 0; kk < BK; kk += WGMMA_K)
        wgmma_m64n256k16(d, &wgmma_sA[kk], &Bs[idx * BK * BN + kk]);
      wgmma_commit();
      wgmma_wait<0>();

      barrier::arrival_token _ = empty[idx].arrive();
    }

    // epilogue -- same fragment layout as m64n128k16, extended to 16 column chunks.
    // Layout (see kernel2 for the derivation): warp w owns rows 16w..16w+15 within
    // the 64-row tile; lane/4 selects the row pair and lane%4 picks a column pair.
    const int lane = threadIdx.x % WARP_SIZE, warp = tid / WARP_SIZE;
    const int row = warp * 16 + lane / 4;

    bf16* tile = C + (size_t)block_n * BN * M + block_m * BM + mi * WGMMA_M;
    auto put = [&](int r, int c, float v) {
      tile[(size_t)c * M + r] = __float2bfloat16(v);
    };
    for (int j = 0; j < WGMMA_N / 16; ++j) {
      int col = 16 * j + 2 * (lane % 4);
      put(row,     col,     d[j][0]);  put(row,     col + 1, d[j][1]);
      put(row + 8, col,     d[j][2]);  put(row + 8, col + 1, d[j][3]);
      put(row,     col + 8, d[j][4]);  put(row,     col + 9, d[j][5]);
      put(row + 8, col + 8, d[j][6]);  put(row + 8, col + 9, d[j][7]);
    }
  }
}

template <int NUM_SM, int GROUP_M, int GROUP_N>
struct scheduler{
  int cursor;
  int tiles_m;
  int tiles_n;
  int total_tiles; // tile_m * tile_n
  __device__ scheduler(int tiles_m_, int tiles_n_, int block)
      : cursor(block), tiles_m(tiles_m_), tiles_n(tiles_n_), total_tiles(tiles_m_ * tiles_n_) {}
  
  __device__ int next() {
    while(cursor < total_tiles){
      const int linear = cursor;
      cursor += NUM_SM;
      const int group_tiles = GROUP_M * GROUP_N;
      const int group = linear / group_tiles;
      const int offset = linear % group_tiles;
      const int groups_n = cdiv(tiles_n, GROUP_N);
      const int tile_m = (group / groups_n) * GROUP_M + offset / GROUP_N;
      const int tile_n = (group % groups_n) * GROUP_N + offset % GROUP_N;
      if (tile_m < tiles_m && tile_n < tiles_n) return tile_m * tiles_n + tile_n;
    }
    return -1;
  }
};

// Trying to hide store latencies
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM>
__global__ __launch_bounds__(NUM_THREADS)
void kernel6(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA,
             const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_M = 64;
  constexpr int WGMMA_K = 16;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int tiles_m = 16;
  constexpr int tiles_n = 8;
  static_assert(num_consumers == 2, "kernel6 needs 2 consumer warpgroups");
  static_assert(BM == 128 && BN == 256 && BK == 64, "kernel6 tile configuration");
  static_assert(BK % WGMMA_K == 0 && BK * sizeof(bf16) == 128,
                "kernel6 needs 128-byte swizzle rows");

  const int warpgroup_idx = threadIdx.x / 128;
  const int tid = threadIdx.x % 128;
  const int tiles_m_count = M / BM;
  const int tiles_n_count = N / BN;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[QSIZE], empty[QSIZE];
  extern __shared__ __align__(1024) bf16 smem[];
  bf16* As = smem;
  bf16* Bs = smem + QSIZE * BM * BK;

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      init(&full[i], num_consumers * 128 + 1);
      init(&empty[i], num_consumers * 128 + 1);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  scheduler<NUM_SM, tiles_m, tiles_n> schedule(tiles_m_count, tiles_n_count, blockIdx.x);

  if (warpgroup_idx == 0) {
    warpgroup_reg_dealloc<24>();
    if (tid == 0) {
      int qidx = 0;
      for (int tile = schedule.next(); tile >= 0; tile = schedule.next()) {
        const int block_m = tile / tiles_n_count;
        const int block_n = tile % tiles_n_count;
        for (int k0 = 0; k0 < K; k0 += BK) {
          if (qidx == QSIZE) qidx = 0;
          auto token = empty[qidx].arrive();
          empty[qidx].wait(std::move(token));
          cde::cp_async_bulk_tensor_2d_global_to_shared(
              &As[qidx * BM * BK], &mapA, k0, block_m * BM, full[qidx]);
          cde::cp_async_bulk_tensor_2d_global_to_shared(
              &Bs[qidx * BK * BN], &mapB, k0, block_n * BN, full[qidx]);
          barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
              full[qidx], 1, (size_t)(BM * BK + BK * BN) * sizeof(bf16));
          ++qidx;
        }
      }
    }
  } else {
    warpgroup_reg_alloc<240>();
    const int consumer_idx = warpgroup_idx - 1;
    float d[BN / 16][8] = {};

    for (int i = 0; i < QSIZE; ++i) {
      barrier::arrival_token _ = empty[i].arrive();
    }

    int qidx = 0;
    for (int tile = schedule.next(); tile >= 0; tile = schedule.next()) {
      const int block_m = tile / tiles_n_count;
      const int block_n = tile % tiles_n_count;
      for (int k0 = 0; k0 < K; k0 += BK) {
        if (qidx == QSIZE) qidx = 0;
        full[qidx].wait(full[qidx].arrive());

        wgmma_fence();
        bf16* wgmma_sA = As + qidx * BM * BK + consumer_idx * WGMMA_M * BK;
        for (int kk = 0; kk < BK; kk += WGMMA_K) {
          wgmma_m64n256k16(d, &wgmma_sA[kk], &Bs[qidx * BK * BN + kk]);
        }
        wgmma_commit();
        wgmma_wait<0>();
        barrier::arrival_token _ = empty[qidx].arrive();
        ++qidx;
      }

      const int lane = tid % WARP_SIZE;
      const int warp = tid / WARP_SIZE;
      const int row = warp * 16 + lane / 4;
      bf16* tile_c = C + (size_t)block_n * BN * M + block_m * BM + consumer_idx * WGMMA_M;
      for (int j = 0; j < BN / 16; ++j) {
        const int col = 16 * j + 2 * (lane % 4);
        tile_c[(size_t)col * M + row] = __float2bfloat16(d[j][0]);
        tile_c[(size_t)(col + 1) * M + row] = __float2bfloat16(d[j][1]);
        tile_c[(size_t)col * M + row + 8] = __float2bfloat16(d[j][2]);
        tile_c[(size_t)(col + 1) * M + row + 8] = __float2bfloat16(d[j][3]);
        tile_c[(size_t)(col + 8) * M + row] = __float2bfloat16(d[j][4]);
        tile_c[(size_t)(col + 9) * M + row] = __float2bfloat16(d[j][5]);
        tile_c[(size_t)(col + 8) * M + row + 8] = __float2bfloat16(d[j][6]);
        tile_c[(size_t)(col + 9) * M + row + 8] = __float2bfloat16(d[j][7]);
      }
      for (int j = 0; j < BN / 16; ++j)
        for (int r = 0; r < 8; ++r)
          d[j][r] = 0.0f;
    }
  }
}

int main(int argc, char **argv) {
  int m = 8192, n = 8192, k = 8192, iters = 20;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) iters = atoi(argv[4]);

  // kernel4: 128x128 output tile per block, K in slabs of 64, 2 warpgroups
  // (warpgroup 0 = producer issuing TMA, warpgroup 1 = consumer running WGMMA).
  constexpr int BM = 128, BN = 128, BK = 64, NUM_THREADS = 256, QSIZE = 5;
  if (m % BM || n % BN || k % BK) {
    fprintf(stderr, "M,N,K must be multiples of %d,%d,%d\n", BM, BN, BK);
    return 1;
  }
  // kernel5: 128x256 output tile, 3 warpgroups (1 producer + 2 consumers), QSIZE 3.
  constexpr int BM5 = 128, BN5 = 256, BK5 = 64, NUM_THREADS5 = 384, QSIZE5 = 3;
  if (m % BM5 || n % BN5 || k % BK5) {
    fprintf(stderr, "kernel5 needs M,N,K multiples of %d,%d,%d\n", BM5, BN5, BK5);
    return 1;
  }
  constexpr int NUM_SM6 = 128;
  if (m / BM5 < 16 || n / BN5 < 8) {
    fprintf(stderr, "kernel6 needs at least 16x8 output tiles\n");
    return 1;
  }

  // A: M x K row-major, B: N x K row-major, C: M x N column-major (see kernel2 header)
  size_t szA = (size_t)m * k, szB = (size_t)n * k, szC = (size_t)m * n;
  bf16 *hA = (bf16 *)malloc(szA * sizeof(bf16));
  bf16 *hB = (bf16 *)malloc(szB * sizeof(bf16));
  bf16 *hC = (bf16 *)malloc(szC * sizeof(bf16));
  for (size_t i = 0; i < szA; i++) hA[i] = __float2bfloat16((rand() % 200 - 100) / 100.f);
  for (size_t i = 0; i < szB; i++) hB[i] = __float2bfloat16((rand() % 200 - 100) / 100.f);

  bf16 *A, *B, *C;
  cudaMalloc(&A, szA * sizeof(bf16));
  cudaMalloc(&B, szB * sizeof(bf16));
  cudaMalloc(&C, szC * sizeof(bf16));
  cudaMemcpy(A, hA, szA * sizeof(bf16), cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB, szB * sizeof(bf16), cudaMemcpyHostToDevice);

  // Tensor maps are built once on the host and passed by value (__grid_constant__).
  CUtensorMap mapA = make_tensor_map<BM, BK>(A, m, k);
  CUtensorMap mapB = make_tensor_map<BN, BK>(B, n, k);
  // kernel4 buffers QSIZE stages of (BM*BK + BK*BN) bf16 = 64 KiB, which exceeds
  // the 48 KiB statically-allocated-__shared__ limit, so stage buffers live in
  // dynamic shared memory and the per-block limit must be raised explicitly.
  constexpr size_t SMEM_BYTES = (size_t)QSIZE * (BM * BK + BK * BN) * sizeof(bf16);
  cudaError_t attr_err = cudaFuncSetAttribute(
      kernel4<BM, BN, BK, NUM_THREADS, QSIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
  if (attr_err != cudaSuccess) {
    fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
    return 1;
  }
  // kernel5's stages are wider (BN = 256): 3 x (128*64 + 64*256) bf16 = 144 KiB.
  constexpr size_t SMEM_BYTES5 = (size_t)QSIZE5 * (BM5 * BK5 + BK5 * BN5) * sizeof(bf16);
  cudaError_t attr_err5 = cudaFuncSetAttribute(
      kernel5<BM5, BN5, BK5, NUM_THREADS5, QSIZE5>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES5);
  if (attr_err5 != cudaSuccess) {
    fprintf(stderr, "cudaFuncSetAttribute(kernel5) failed: %s\n", cudaGetErrorString(attr_err5));
    return 1;
  }
  cudaError_t attr_err6 = cudaFuncSetAttribute(
      kernel6<BM5, BN5, BK5, NUM_THREADS5, QSIZE5, NUM_SM6>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES5);
  if (attr_err6 != cudaSuccess) {
    fprintf(stderr, "cudaFuncSetAttribute(kernel6) failed: %s\n", cudaGetErrorString(attr_err6));
    return 1;
  }
  // kernel5 stages B in 256-row boxes, so it needs its own (wider) tensor map.
  CUtensorMap mapB5 = make_tensor_map<BN5, BK5>(B, n, k);
  // How many blocks actually fit on one SM, and how many of their warps issue WGMMA.
  // kernel4 spends a whole warpgroup on TMA, so resident warps != working warps.
  auto occupancy = [](const char* name, auto kern, int threads, size_t dyn_smem, int producer_warps) {
    int blocks = 0, regs = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, threads, dyn_smem);
    cudaFuncAttributes fa{};
    cudaFuncGetAttributes(&fa, kern);
    regs = fa.numRegs;
    int warps = blocks * threads / WARP_SIZE;
    printf("%-8s %d regs/thread  %d blocks/SM  %d warps/SM  %d of them issuing WGMMA\n",
           name, regs, blocks, warps, warps - blocks * producer_warps);
  };
  occupancy("kernel3", kernel3<BM, BN, BK, 128>, 128, 0, 0);
  occupancy("kernel4", kernel4<BM, BN, BK, NUM_THREADS, QSIZE>, NUM_THREADS, SMEM_BYTES, 4);
  occupancy("kernel5", kernel5<BM5, BN5, BK5, NUM_THREADS5, QSIZE5>, NUM_THREADS5, SMEM_BYTES5, 4);
  occupancy("kernel6", kernel6<BM5, BN5, BK5, NUM_THREADS5, QSIZE5, NUM_SM6>, NUM_THREADS5,
            SMEM_BYTES5, 4);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm up, spot-check 256 random entries against a CPU dot product, then time `iters` runs.
  auto bench = [&](const char* name, auto&& launch) {
    cudaMemset(C, 0, szC * sizeof(bf16));
    launch();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "%s failed: %s\n", name, cudaGetErrorString(err));
      return 0.0;
    }
    cudaMemcpy(hC, C, szC * sizeof(bf16), cudaMemcpyDeviceToHost);
    for (int s = 0; s < 256; s++) {
      int i = rand() % m, j = rand() % n;
      double ref = 0;
      for (int kk = 0; kk < k; kk++)
        ref += (double)__bfloat162float(hA[(size_t)i * k + kk]) * __bfloat162float(hB[(size_t)j * k + kk]);
      float got = __bfloat162float(hC[(size_t)j * m + i]);
      if (fabs(ref - got) > 1e-2 * (1 + fabs(ref))) {
        fprintf(stderr, "%s MISMATCH C[%d][%d] = %f, expected %f\n", name, i, j, got, ref);
        return 0.0;
      }
    }

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = 2.0 * m * n * k * iters / (ms * 1e6);
    printf("%-8s M=%d N=%d K=%d  %.3f ms/iter  %.2f GFLOPs/s\n", name, m, n, k, ms / iters, gflops);
    return gflops;
  };

  // kernel3 is one warpgroup with static smem, so no launch attribute is needed.
  double g3 = bench("kernel3", [&] {
    kernel3<BM, BN, BK, 128><<<(m / BM) * (n / BN), 128>>>(C, m, n, k, mapA, mapB);
  });
  double g4 = bench("kernel4", [&] {
    kernel4<BM, BN, BK, NUM_THREADS, QSIZE><<<(m / BM) * (n / BN), NUM_THREADS, SMEM_BYTES>>>(
        C, m, n, k, mapA, mapB);
  });
  double g5 = bench("kernel5", [&] {
    kernel5<BM5, BN5, BK5, NUM_THREADS5, QSIZE5>
        <<<(m / BM5) * (n / BN5), NUM_THREADS5, SMEM_BYTES5>>>(
            C, m, n, k, mapA, mapB5);
  });
  double g6 = bench("kernel6", [&] {
    kernel6<BM5, BN5, BK5, NUM_THREADS5, QSIZE5, NUM_SM6>
        <<<NUM_SM6, NUM_THREADS5, SMEM_BYTES5>>>(C, m, n, k, mapA, mapB5);
  });

  // The bf16 baseline.  Our layout is A row-major MxK, B row-major NxK, C col-major MxN;
  // in cuBLAS's column-major world that is A^T (ld=K) times B (ld=K) -- i.e. a TN gemm,
  // which is also cuBLAS's fastest tensor-core path.
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.f, beta = 0.f;
  double gcb = bench("cublas", [&] {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                 &alpha, A, CUDA_R_16BF, k, B, CUDA_R_16BF, k,
                 &beta, C, CUDA_R_16BF, m,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  });
  cublasDestroy(handle);

  if (g3 == 0 || g4 == 0 || g5 == 0 || g6 == 0 || gcb == 0) return 1;
  printf("kernel3 %.1f%% of cuBLAS   kernel4 %.1f%% of cuBLAS   kernel5 %.1f%% of cuBLAS   kernel6 %.1f%% of cuBLAS\n",
         100 * g3 / gcb, 100 * g4 / gcb, 100 * g5 / gcb, 100 * g6 / gcb);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(hA);
  free(hB);
  free(hC);
  return 0;
}
