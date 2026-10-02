cublass (TF32 tensor cores)
M=8192 N=8192 K=8192  2.422 ms/iter  453989.75 GFLOPs/s
kernel1 -> going for shared memory 
warp tiling 
M=8192 N=8192 K=8192  42.921 ms/iter  25617.24 GFLOPs/s
kernel2 -> using Tensor cores and TMA 
M=8192 N=8192 K=8192  4.076 ms/iter  269737.90 GFLOPs/s
kernel3 -> bigger the output tiles 
M=8192 N=8192 K=8192  2.054 ms/iter  435709.33 GFLOPs/s 
kernel4 -> using producers and consumers
M=8192 N=8192 K=8192  2.671 ms/iter  520909.13 GFLOPs/s
kernel5 -> we pushed the tile size with kernel4 logic
M=8192 N=8192 K=8192  2.671 ms/iter  594019.80 GFLOPs/s
interestingly we are using setmaxnreg from PTX in kernel5 but still getting around ~75% we tried kernel5 without them we got around ~78%
kernel6 -> it is a mixture of hiding store latencies and added custom PTX barrier 
M=8192 N=8192 K=8192  1.500 ms/iter  733102.37 GFLOPs/s
kernel7 -> use cluster and TMA multicast 
M=8192 N=8192 K=8192  1.435 ms/iter  766025.73 GFLOPs/s
warp tiling, beyond the algorithm:
- warp sub-tile iteration 2x2 so the 8x4 lane grid actually covers the 64x32 warp tile
- As[BM][BK+1] padding -> no 8-way bank conflict on A fragment loads
- no #pragma unroll on the k loop: 18.5 -> 25.9 TFLOP/s (full unroll spilled)
- __launch_bounds__(256), __restrict__, tile shape as template ints (index math folds)
- flat coalesced staging loop, zero-fill at edges so the FFMA loop has no bounds checks
- 2-D grid (1-D grid + blockIdx.y was computing 1/128 of C)

next (measured 45.2 TFLOP/s, 2.44x): transpose A into As[k][m], float4 loads/stores, double-buffered smem

animation: warp_tiling_anim/index.html   interactive explorer: warp_tiling_anim/explorer.html
