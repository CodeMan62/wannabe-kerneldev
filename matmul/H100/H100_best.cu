// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -lcuda -o matmul kernel.cu
// (H100 only: wgmma/TMA need the sm_90a target; -lcuda for cuTensorMapEncodeTiled)
// becoming a god in B200 kernel writing
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
// A block owns BM x BN, but each WGMMA is only WGMMA_M x WGMMA_N x WGMMA_K.
// The loops over the block's M and N subtile coordinates are deliberately explicit:
//
//   [BM/WGMMA_M, BK/WGMMA_K] * [BK/WGMMA_K, BN/WGMMA_N]
//
// This keeps one CTA (and one pair of TMA stages) while exposing the cost of doing
// the decomposition itself.  Accumulators for all M/N subtiles remain live until the
// epilogue, so the K loop is shared by every micro-matmul rather than reloading A/B.
// -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void kernel3(bf16* __restrict__ C, int M, int N, int K,
             const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB) {
  constexpr int WGMMA_M = 64, WGMMA_N = 64, WGMMA_K = 16;
  static_assert(NUM_THREADS == 128, "kernel3 uses one warpgroup");
  static_assert(BM % WGMMA_M == 0 && BN % WGMMA_N == 0,
                "block dimensions must be multiples of the WGMMA tile");
  static_assert(BK % WGMMA_K == 0 && BK * sizeof(bf16) == 128,
                "BK must be WGMMA-compatible and one 128B swizzle row");

  constexpr int MTILES = BM / WGMMA_M;
  constexpr int NTILES = BN / WGMMA_N;
  __shared__ alignas(1024) bf16 sA[BM * BK];
  __shared__ alignas(1024) bf16 sB[BN * BK];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier barA, barB;

  // d[mi][nj] is the 32-float fragment owned by this thread for one micro-matmul.
  float d[MTILES][NTILES][4][8] = {};
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
    // The nested loops are the decomposition under test: each call is one
    // 64x64x16 regular matrix multiply, while all calls share this K slab.
    for (int mi = 0; mi < MTILES; ++mi)
      for (int nj = 0; nj < NTILES; ++nj)
        for (int kk = 0; kk < BK; kk += WGMMA_K)
          wgmma_m64n64k16(d[mi][nj], &sA[mi * WGMMA_M * BK + kk],
                          &sB[nj * WGMMA_N * BK + kk]);
    wgmma_commit();
    wgmma_wait<0>();
  }

  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const int warp_row = warp * 16 + lane / 4;        // row within a 64x64 sub-tile
  bf16* tile = C + (size_t)block_n * BN * M + block_m * BM;
  auto put = [&](int r, int c, float v) { tile[(size_t)c * M + r] = __float2bfloat16(v); };
  for (int mi = 0; mi < MTILES; ++mi)
    for (int nj = 0; nj < NTILES; ++nj)
      for (int j = 0; j < WGMMA_N / 16; ++j) {
        int row = mi * WGMMA_M + warp_row;
        int col = nj * WGMMA_N + 16 * j + 2 * (lane % 4);
        put(row,     col,     d[mi][nj][j][0]);  put(row,     col + 1, d[mi][nj][j][1]);
        put(row + 8, col,     d[mi][nj][j][2]);  put(row + 8, col + 1, d[mi][nj][j][3]);
        put(row,     col + 8, d[mi][nj][j][4]);  put(row,     col + 9, d[mi][nj][j][5]);
        put(row + 8, col + 8, d[mi][nj][j][6]);  put(row + 8, col + 9, d[mi][nj][j][7]);
      }
}


int main(int argc, char **argv) {
  int m = 8192, n = 8192, k = 8192, iters = 20;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) iters = atoi(argv[4]);

  // 128x128 output tile per block, decomposed into 64x64 WGMMA tiles; K in slabs of 64.
  constexpr int BM = 128, BN = 128, BK = 64, NUM_THREADS = 128;
  if (m % BM || n % BN || k % BK) {
    fprintf(stderr, "M,N,K must be multiples of %d,%d,%d\n", BM, BN, BK);
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
  auto launch = [&] {
    kernel3<BM, BN, BK, NUM_THREADS><<<(m / BM) * (n / BN), NUM_THREADS>>>(C, m, n, k, mapA, mapB);
  };

  // Warm-up + correctness: spot-check 256 random entries against a CPU dot product.
  launch();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaMemcpy(hC, C, szC * sizeof(bf16), cudaMemcpyDeviceToHost);
  for (int s = 0; s < 256; s++) {
    int i = rand() % m, j = rand() % n;
    double ref = 0;
    for (int kk = 0; kk < k; kk++)
      ref += (double)__bfloat162float(hA[(size_t)i * k + kk]) * __bfloat162float(hB[(size_t)j * k + kk]);
    float got = __bfloat162float(hC[(size_t)j * m + i]);
    if (fabs(ref - got) > 1e-2 * (1 + fabs(ref))) {
      fprintf(stderr, "MISMATCH C[%d][%d] = %f, expected %f\n", i, j, got, ref);
      return 1;
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * m * n * k * iters / (ms * 1e6);
  printf("M=%d N=%d K=%d  %.3f ms/iter  %.2f GFLOPs/s\n", m, n, k, ms / iters, gflops);

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
