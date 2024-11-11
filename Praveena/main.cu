#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
 
// Kernel to square each element
__global__ void squareElements(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}
 
// calculate the mean of the squared elements
__global__ void calculateMean(const float* input, float* output, size_t size) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    shared_data[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
 
    if (tid == 0) {
        atomicAdd(output, shared_data[0] / size);
    }
}
 
// calculate the square root of the mean
__global__ void calculateSqrt(float* mean) {
    *mean = sqrt(*mean);
}
 
// normalize the input array with the RMS value
__global__ void normalize(const float* input, float* output, float rms, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / rms;
    }
}
 
// Function to compare arrays within a given tolerance
bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
 
int main() {
    // Initialize the tensor with random values for demonstration
    torch::manual_seed(42);  // Set seed for reproducibility
    torch::Tensor input = torch::randn({1024, 1024, 32});  // Random tensor with shape (1024, 1024, 32)
    std::cout << "Original Tensor Shape: " << input.sizes() << std::endl;
 
    // Host input pointer to flattened array
    float* h_ptr = input.data_ptr<float>();
 
    // CPU RMS normalization
    auto tensor_squared = input.pow(2);
    auto mean_squared = tensor_squared.mean();
    auto rms = mean_squared.sqrt();
    torch::Tensor output = input / rms;
    float* h_output_ptr = output.data_ptr<float>();
 
    // GPU allocation and data copy
    size_t output_size = 1;
    for (int i : output.sizes()) output_size *= i;
 
    float *d_input, *d_squared, *d_mean, *d_output;
 
    // Allocate memory on GPU
    cudaMalloc(&d_input, output_size * sizeof(float));
    cudaMalloc(&d_squared, output_size * sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
 
    cudaMemcpy(d_input, h_ptr, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_mean, 0, sizeof(float));
 
    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (output_size + blockSize - 1) / blockSize;
 
    // Squaring each element
    squareElements<<<gridSize, blockSize>>>(d_input, d_squared, output_size);
 
    // Calculating the mean of squared elements
    calculateMean<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_squared, d_mean, output_size);
 
    // Calculating the square root of the mean
    calculateSqrt<<<1, 1>>>(d_mean);
 
    // Normalizing the input by RMS
    float* h_rms = new float[1];
    cudaMemcpy(h_rms, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    normalize<<<gridSize, blockSize>>>(d_input, d_output, *h_rms, output_size);
 
    // Copy result back to host
    float* cuda_output = new float[output_size];
    cudaMemcpy(cuda_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
 
    // Compare CPU and GPU outputs
    if (compareArrays(h_output_ptr, cuda_output, output_size, 1e-3)) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }
 
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_squared);
    cudaFree(d_mean);
    cudaFree(d_output);
    delete[] cuda_output;
    delete[] h_rms;
 
    return 0;
}