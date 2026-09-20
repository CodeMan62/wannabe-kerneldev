// nvcc -O3 -arch=sm_86 -o matmul3050 kernel.cu
// becoming a god in B200 kernel writing 
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
//#include <cmath>
constexpr int WARP_SIZE = 32;

__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
__global__ void kernel1(const float *A, const float *B, float *C, int M, int N,
                        int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float tmp = 0.f;
    for (int i = 0; i < K; ++i)
      tmp += A[row * K + i] * B[i * N + col];
    C[row * N + col] = tmp;
  }
}
// 2D block shared memory cache blocking
template<int BLOCKSIZE>
__global__ void kernel2(const float* A, const float* B, float *C, int M, int N, int K){
  const int thread_row = threadIdx.y;
  const int thread_col = threadIdx.x;
  const int blk_row = blockIdx.y;
  const int blk_col = blockIdx.x; 
  A += blk_row * BLOCKSIZE * K;
  B += blk_col * BLOCKSIZE;
  C += blk_row * BLOCKSIZE * N + blk_col * BLOCKSIZE;
  __shared__ float As[BLOCKSIZE* BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE* BLOCKSIZE];

  float tmp = 0.0f;
  for(int blkIdx = 0; blkIdx < K; blkIdx += BLOCKSIZE){
    As[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
    Bs[thread_row * BLOCKSIZE + thread_col] = A[thread_row * N + thread_col];
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
    for (int i=0;i<BLOCKSIZE;i++){
      tmp += As[thread_row * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + thread_col];
    }
    __syncthreads();
  }
  C[thread_row * N + thread_col] = tmp;
}
// we want to load a (HEIGHT, WIDTH) tile from global to shared memory.
// just load a BLOCK_SIZE of data until the whole tile is loaded.
template <int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                           float out[HEIGHT][WIDTH], int tid) {
  for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    out[row][col] = row < in_max_row && col < in_max_col ? in[row * in_row_stride + col] : 0.0f;
  }
}
// 2D thread tiling
// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void kernel3(const float* A, const float* B, float* C, int M, int N, int K){
//   constexpr int BLOCKSIZE = (BM * BN) / (TM * TN);
//   const int tid = threadIdx.x;
//   const int block_id = blockIdx.x;

//   const int num_blocks_per_row = cdiv(N, BN);
//   const int block_id_m = block_id / num_blocks_per_row;
//   const int block_id_n = block_id % num_blocks_per_row;
//   const int offset_m = block_id_m * BM;
//   const int offset_n = block_id_n * BN;

//   const int num_threads_tiles_per_row = BN/ TN;
//   const int thread_tile_id_m = tid / num_threads_tiles_per_row;
//   const int thread_tile_id_n = tid % num_threads_tiles_per_row;
//   const int thread_tile_offset_m = thread_tile_id_m * TM;
//   const int thread_tile_offset_n = thread_tile_id_n * TN;

//   __shared__ float As[BM][BK];
//   __shared__ float Bs[BK][BN];
//   float acc[TM][TN] = {0.0f};
//   A += offset_m * K; 
//   B += offset_n;
//   const float *A_thread_tile = reinterpret_cast<const float *>(As) + thread_tile_offset_m * BK;
//   const float *B_thread_tile = reinterpret_cast<const float *>(Bs) + thread_tile_offset_n;
//   for(int offset_k = 0; offset_k < K; offset_k +=BN){
//     load_shmem<BLOCKSIZE, BM, BK>(A, K, M - offset_m, K - offset_k, As, tid);
//     load_shmem<BLOCKSIZE, BK, BN>(A, N, K - offset_k, N - offset_n, Bs, tid);
//     __syncthreads();
//     for(int k = 0;k<BK;k++){
//       float A_regg[TM];
//       float B_regg[TN];
//       for(int m = 0;m<TM;m++){
//         A_regg[m] = A_thread_tile[m * BK+k];
//       }
//       for(int n = 0;n<TN;n++){
//         B_regg[n] = B_thread_tile[k*BN+n];
//       }
//       for(int m = 0;m<TM;m++){
//         for(int n = 0;n<TN;n++){
//           acc[m][n] += A_regg[m] * B_regg[n];
//         }
//       }
//     }
//     __syncthreads();
//     A += BK;
//     B += BK * N;
//   }
//   C += (offset_m + thread_tile_offset_m) * N + (offset_n + thread_tile_offset_n);
//   for (int m = 0; m < TM; m++)
//   for (int n = 0; n < TN; n++)
//     if (m < (M - (offset_m + thread_tile_offset_m)) && n < (N - (offset_n + thread_tile_offset_n)))
//       C[m * N + n] = acc[m][n];
// }
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void kernel3(const float *A, const float *B, float *C, int M, int N, int K) {
  constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;

  const int num_blocks_per_row = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  const int num_thread_tiles_per_row = BLOCK_N / THREAD_N;
  const int thread_tile_id_m = tid / num_thread_tiles_per_row;
  const int thread_tile_id_n = tid % num_thread_tiles_per_row;
  const int thread_tile_offset_m = thread_tile_id_m * THREAD_M;
  const int thread_tile_offset_n = thread_tile_id_n * THREAD_N;

  __shared__ float A_shmem[BLOCK_M][BLOCK_K];
  __shared__ float B_shmem[BLOCK_K][BLOCK_N];
  float acc[THREAD_M][THREAD_N] = {0.0f};

  A += offset_m * K;
  B += offset_n;
  const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + thread_tile_offset_m * BLOCK_K;
  const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + thread_tile_offset_n;

  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
    load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
    load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
    __syncthreads();

    // mini-matmul with thread-tile. same structure as block-tile.
    // THREAD_K = 1
    for (int k = 0; k < BLOCK_K; k++) {
      float A_reg[THREAD_M];  // register cache
      float B_reg[THREAD_N];

      // load data from shared memory to registers
      // there is shared memory bank conflict
      for (int m = 0; m < THREAD_M; m++)
        A_reg[m] = A_thread_tile[m * BLOCK_K + k];

      for (int n = 0; n < THREAD_N; n++)
        B_reg[n] = B_thread_tile[k * BLOCK_N + n];

      // for each (THREAD_M, THEAD_N) output, we only need to read
      // (THREAD_M, BLOCK_K) of A and (BLOCK_K, THREAD_N) for B from shared memory.
      for (int m = 0; m < THREAD_M; m++)
        for (int n = 0; n < THREAD_N; n++)
          acc[m][n] += A_reg[m] * B_reg[n];
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  C += (offset_m + thread_tile_offset_m) * N + (offset_n + thread_tile_offset_n);

  // uncoalesced memory write
  // fixing it doesn't seem to make the kernel faster.
  // vectorized write is slower.
  for (int m = 0; m < THREAD_M; m++)
    for (int n = 0; n < THREAD_N; n++)
      if (m < (M - (offset_m + thread_tile_offset_m)) && n < (N - (offset_n + thread_tile_offset_n)))
        C[m * N + n] = acc[m][n];
}

// warp tiling
template <int BM, int BN, int BK, int WM, int WN, int MMA_M, int MMA_N, int TN>
__global__ void kernel4(const float *A, const float *B, float *C, int M, int N, int K){
  constexpr int BLOCKSIZE = (BM * BN) / (WM*WN) * WARP_SIZE;
  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_blocks_per_row = cdiv(N, BN);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BM;
  const int offset_n = block_id_n * BN;
  constexpr int NUM_MMA_M = WM / MMA_M;
  constexpr int NUM_MMA_N = WN / MMA_N;
  constexpr int THREAD_M = MMA_M * MMA_N / (WARP_SIZE * TN);
  constexpr int num_warps_per_row = BN / WN;
  const int warp_id = tid / WARP_SIZE;
  const int warp_id_m = warp_id / num_warps_per_row;
  const int warp_id_n = warp_id % num_warps_per_row;
  const int warp_tile_offset_m = warp_id_m * WM;
  const int warp_tile_offset_n = warp_id_n * WN;
  const int lane_id = tid % WARP_SIZE;
  constexpr int num_thread_tiles_per_row = MMA_N / TN;
  const int thread_tile_id_m = lane_id / num_thread_tiles_per_row;
  const int thread_tile_id_n = lane_id % num_thread_tiles_per_row;
  const int thread_tile_offset_m = thread_tile_id_m * THREAD_M;
  const int thread_tile_offset_n = thread_tile_id_n * TN;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][TN] = {0.0f};
  // points to the corresponding thread tile of the current thread in the first MMA tile
  const float *A_thread_tile = reinterpret_cast<const float *>(As) + (warp_tile_offset_m + thread_tile_offset_m) * BK;
  const float *B_thread_tile = reinterpret_cast<const float *>(Bs) + (warp_tile_offset_n + thread_tile_offset_n);
  // outer loop K 
  for (int offset_k = 0; offset_k < K; offset_k += BK){
    load_shmem<BLOCKSIZE, BM, BK>(A, K, M - offset_m, K - offset_k, As, tid);
    load_shmem<BLOCKSIZE, BK, BN>(B, N, K - offset_k, N - offset_n, Bs, tid);
    __syncthreads();
    for(int k = 0;k<BK;k++){
      float A_reg[NUM_MMA_M][THREAD_M];
      float B_reg[NUM_MMA_N][TN];
      for (int mma_tile_id_m = 0;mma_tile_id_m<NUM_MMA_M;mma_tile_id_m++){
        for (int tm = 0;tm < THREAD_M;tm++){
          A_reg[mma_tile_id_m][tm] = A_thread_tile[(mma_tile_id_m * MMA_M + tm) * BK + k];
        }
      }
      for (int mma_tile_id_n = 0;mma_tile_id_n<NUM_MMA_N;mma_tile_id_n++){
        for (int tn = 0;tn < THREAD_M;tn++){
          A_reg[mma_tile_id_n][tn] = B_thread_tile[k * BK + (mma_tile_id_n* MMA_N + tn)];
        }
      }
      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
          for (int tm = 0; tm < THREAD_M; tm++)
            for (int tn = 0; tn < TN; tn++)
              acc[mma_tile_id_m][mma_tile_id_n][tm][tn] += A_reg[mma_tile_id_m][tm] * B_reg[mma_tile_id_n][tn];
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }
  const int C_offset_m = offset_m + warp_tile_offset_m + thread_tile_offset_m;
  const int C_offset_n = offset_n + warp_tile_offset_n + thread_tile_offset_n;
  C += C_offset_m * N + C_offset_n;
}

template <int BLOCK_SIZE, int HEIGHT, int WIDTH, bool TRANSPOSED>
__device__ void load_shmem_vectorized(const float *in, int in_row_stride, float *out, int tid) {
  for (int offset = 0; offset < HEIGHT * WIDTH; offset += BLOCK_SIZE * 4) {
    const int idx = offset + tid * 4;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    float4 tmp = reinterpret_cast<const float4 *>(&in[row * in_row_stride + col])[0];

    if (TRANSPOSED) {
      out[(col + 0) * HEIGHT + row] = tmp.x;
      out[(col + 1) * HEIGHT + row] = tmp.y;
      out[(col + 2) * HEIGHT + row] = tmp.z;
      out[(col + 3) * HEIGHT + row] = tmp.w;
    } else
      reinterpret_cast<float4 *>(&out[row * WIDTH + col])[0] = tmp;
  }
}

// vectorized memory access without bounds check
// only memory access is different from v5
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int THREAD_N, bool TRANSPOSE_A_shmem>
__global__ void kernel5(const float *A, const float *B, float *C, int M, int N, int K) {
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(WARP_M % MMA_M == 0);
  static_assert(WARP_N % MMA_N == 0);
  static_assert((MMA_M * MMA_N / THREAD_N) % WARP_SIZE == 0);
  static_assert(THREAD_N % 4 == 0);  // so we can use vectorized access
  constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
  constexpr int NUM_MMA_M = WARP_M / MMA_M;
  constexpr int NUM_MMA_N = WARP_N / MMA_N;
  constexpr int THREAD_M = MMA_M * MMA_N / (WARP_SIZE * THREAD_N);

  const int tid = threadIdx.x;
  const int block_id = blockIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_blocks_per_row = cdiv(N, BLOCK_N);
  const int block_id_m = block_id / num_blocks_per_row;
  const int block_id_n = block_id % num_blocks_per_row;
  const int offset_m = block_id_m * BLOCK_M;
  const int offset_n = block_id_n * BLOCK_N;

  constexpr int num_warps_per_row = BLOCK_N / WARP_N;
  const int warp_id_m = warp_id / num_warps_per_row;
  const int warp_id_n = warp_id % num_warps_per_row;
  const int warp_tile_offset_m = warp_id_m * WARP_M;
  const int warp_tile_offset_n = warp_id_n * WARP_N;

  constexpr int num_thread_tiles_per_row = MMA_N / THREAD_N;
  const int thread_tile_id_m = lane_id / num_thread_tiles_per_row;
  const int thread_tile_id_n = lane_id % num_thread_tiles_per_row;
  const int thread_tile_offset_m = thread_tile_id_m * THREAD_M;
  const int thread_tile_offset_n = thread_tile_id_n * THREAD_N;

  A += offset_m * K;
  B += offset_n;

  __shared__ float A_shmem[BLOCK_M * BLOCK_K];
  __shared__ float B_shmem[BLOCK_K * BLOCK_N];
  float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][THREAD_N] = {0.0f};

  const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + (warp_tile_offset_m + thread_tile_offset_m) * (TRANSPOSE_A_shmem ? 1 : BLOCK_K);
  const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + (warp_tile_offset_n + thread_tile_offset_n);

  for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
    load_shmem_vectorized<BLOCK_SIZE, BLOCK_M, BLOCK_K, TRANSPOSE_A_shmem>(A, K, A_shmem, tid);
    load_shmem_vectorized<BLOCK_SIZE, BLOCK_K, BLOCK_N, false>(B, N, B_shmem, tid);
    __syncthreads();

    for (int k = 0; k < BLOCK_K; k++) {
      float A_reg[NUM_MMA_M][THREAD_M];
      float B_reg[NUM_MMA_N][THREAD_N];

      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        if (TRANSPOSE_A_shmem) {
          static_assert(THREAD_M % 4 == 0);
          for (int tm = 0; tm < THREAD_M; tm += 4) {
            float4 tmp = reinterpret_cast<const float4 *>(&A_thread_tile[k * BLOCK_M + (mma_tile_id_m * MMA_M + tm)])[0];
            reinterpret_cast<float4 *>(&A_reg[mma_tile_id_m][tm])[0] = tmp;
          }
        }
        else {
          for (int tm = 0; tm < THREAD_M; tm++)
            A_reg[mma_tile_id_m][tm] = A_thread_tile[(mma_tile_id_m * MMA_M + tm) * BLOCK_K + k];
        }

      for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
        for (int tn = 0; tn < THREAD_N; tn += 4) {
          float4 tmp = reinterpret_cast<const float4 *>(&B_thread_tile[k * BLOCK_N + (mma_tile_id_n * MMA_N + tn)])[0];
          reinterpret_cast<float4 *>(&B_reg[mma_tile_id_n][tn])[0] = tmp;
        }

      for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
        for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
          for (int tm = 0; tm < THREAD_M; tm++)
            for (int tn = 0; tn < THREAD_N; tn++)
              acc[mma_tile_id_m][mma_tile_id_n][tm][tn] += A_reg[mma_tile_id_m][tm] * B_reg[mma_tile_id_n][tn];
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K * N;
  }

  const int C_offset_m = offset_m + warp_tile_offset_m + thread_tile_offset_m;
  const int C_offset_n = offset_n + warp_tile_offset_n + thread_tile_offset_n;
  C += C_offset_m * N + C_offset_n;

  for (int mma_m = 0; mma_m < WARP_M; mma_m += MMA_M)
    for (int mma_n = 0; mma_n < WARP_N; mma_n += MMA_N)
      for (int tm = 0; tm < THREAD_M; tm++)
        for (int tn = 0; tn < THREAD_N; tn += 4) {
          const float4 tmp = reinterpret_cast<const float4 *>(&acc[mma_m / MMA_M][mma_n / MMA_N][tm][tn])[0];
          reinterpret_cast<float4 *>(&C[(mma_m + tm) * N + (mma_n + tn)])[0] = tmp;
        }
}

int main(int argc, char **argv) {
  int m = 8192, n = 8192, k = 8192, iters = 20;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) iters = atoi(argv[4]);

  float *A, *B, *C;
  cudaMalloc(&A, (size_t)m * k * sizeof(float));
  cudaMalloc(&B, (size_t)k * n * sizeof(float));
  cudaMalloc(&C, (size_t)m * n * sizeof(float));

  //const int BM = 128, BN = 128, BK=32;
  //const int TM = 8, TN = 8;
  //constexpr int BLOCKSIZE = (BM * BN) / (TM * TN);
  //dim3 block(32, 32);
  //dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  //const int grid_size = cdiv(m*n, BM*BN);

  //kernel3<BM, BN, BK, TM, TN><<<grid_size, BLOCKSIZE>>>(A, B, C, m, n, k);
  // kernel 4
  //const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
  //const int WARP_M = 32, WARP_N = 32;
  //const int MMA_M = 16, MMA_N = 32;
  //const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 4

  //const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 256
  //const int grid_size = cdiv(m * n, BLOCK_M * BLOCK_N);
  //kernel4<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, m, n, k);
  const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;
  const int WARP_M = 64, WARP_N = 64;
  const int MMA_M = 16, MMA_N = 32;
  const int THREAD_N = 4;  // THREAD_M = MMA_M * MMA_N / 32 / THREAD_N = 4

  const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;  // 128
  const int grid_size = cdiv(m * n, BLOCK_M * BLOCK_N);
  kernel5<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N, false><<<grid_size, BLOCK_SIZE>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    kernel4<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, m, n, k);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * m * n * k * iters / (ms * 1e6);
  printf("M=%d N=%d K=%d  %.3f ms/iter  %.2f GFLOPs/s\n", m, n, k, ms / iters,
         gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
