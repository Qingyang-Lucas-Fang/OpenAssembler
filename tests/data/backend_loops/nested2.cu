extern "C" __global__ void nested2(const int* a, int* out, int n, int m){
  int t = threadIdx.x; int acc = 0;
  for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) acc += a[(t + i*m + j) & 63] * (i + 1) - j;
  out[t] = acc;
}
