extern "C" __global__ void barrier64(const float* input, float* output) {
    __shared__ float tile[64];
    unsigned t = threadIdx.x;
    tile[t] = input[t];
    __syncthreads();
    output[t] = tile[63 - t];
}
