cublass 
M=8192 N=8192 K=8192  1.062 ms/iter  1034860.20 GFLOPs/s
kernel1 -> vectorized warp tiling because we know this is on top for this 
M=8192 N=8192 K=8192  5.219 ms/iter  210663.92 GFLOPs/s
kernel2 -> let's find out first read the B200 GPU I guess
