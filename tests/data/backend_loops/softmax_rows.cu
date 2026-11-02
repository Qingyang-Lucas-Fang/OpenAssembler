extern "C" __global__ void softmax_rows(const float* x, float* y, int n){
  int row = threadIdx.x; const float* r = x + row*n; float m = -3.402823466e38f;
  for (int j = 0; j < n; ++j) m = fmaxf(m, r[j]);
  float s = 0.f;
  for (int j = 0; j < n; ++j) s += expf(r[j] - m);
  y[row] = s;
}
