// nvcc -O3 -arch=sm_90 -lcublas -o h100_cublass h100_cublass.cu
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  float alpha = 1.f, beta = 0.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = 2.0 * m * n * k * iters / (ms * 1e6);
  printf("M=%d N=%d K=%d  %.3f ms/iter  %.2f GFLOPs/s\n", m, n, k, ms / iters, gflops);

  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
