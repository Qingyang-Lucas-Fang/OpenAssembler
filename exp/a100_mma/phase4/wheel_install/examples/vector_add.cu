// Minimal openassembler.cppfront example: CUDA C compiled to PTX by nvcc (--ptx),
// then to SASS by our own backend -- no ptxas code generation involved
// (see README.md "How it works"). extern "C" keeps the kernel's symbol
// name un-mangled so --kernel/--entry can find it by its plain name.
//
//   openassembler compile examples/vector_add.cu -o out.cubin --verify

extern "C" __global__ void vector_add(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
