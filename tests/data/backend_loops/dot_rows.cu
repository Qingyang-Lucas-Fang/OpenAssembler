extern "C" __global__ void dot_rows(float* a, float* b, float* out, int n){ int row=threadIdx.x; float acc=0.f; for(int j=0;j<n;++j) acc += a[row*n+j]*b[row*n+j]; out[row]=acc; }
