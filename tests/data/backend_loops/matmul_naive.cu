extern "C" __global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K){
  int i = blockIdx.y*blockDim.y + threadIdx.y, j = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < M && j < N) { float s = 0.f; for (int k = 0; k < K; ++k) s += A[i*K+k]*B[k*N+j]; C[i*N+j] = s; }
}
