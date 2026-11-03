#include <cuda_runtime.h>
#include <vector>

__global__ void reduceKernel(const float* g_idata, float* g_odata, int N) {
    // Dynamically allocated shared memory for block-wise reduction
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Perform first level of addition during loading, reducing global memory reads
    float mySum = 0.0f;
    if (i < N) {
        mySum += g_idata[i];
    }
    if (i + blockDim.x < N) {
        mySum += g_idata[i + blockDim.x];
    }
    
    sdata[tid] = mySum;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    // Using grid-stride / block-reduction sizing: each block handles up to threadsPerBlock * 2 elements
    int blocksPerGrid = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    if (blocksPerGrid == 0) blocksPerGrid = 1;

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch first reduction kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);

    // If there are multiple blocks, we can handle the final reduction on host or with a secondary launch.
    // For simplicity and correctness with arbitrary N, we can copy partial sums back or reduce them.
    std::vector<float> h_output(blocksPerGrid);
    cudaMemcpy(h_output.data(), d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        total_sum += h_output[i];
    }

    *output = total_sum;

    cudaFree(d_input);
    cudaFree(d_output);
}