#include <torch/torch.h>
#include <iostream>
#include "cuda_runtime.h"
#include <cmath>
using namespace std;

// Kernel Code
__global__ void softmaxKernel(float* input, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;

    if (idx < total_elements) {
        int b = idx / spatial_size; 
        int hw = idx % spatial_size; // Spatial position within height * width

        int base_idx = b * channels * spatial_size + hw;

        //finding max
        float max_val = input[base_idx];
        for (int c = 1; c < channels; ++c) {
            max_val = fmaxf(max_val, input[base_idx + c * spatial_size]);
        }

        //Calculating exponentials and their sum
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c) {
            output[base_idx + c * spatial_size] = expf(input[base_idx + c * spatial_size] - max_val);
            sum_exp += output[base_idx + c * spatial_size];
        }

        // Normalization to get softmax
        for (int c = 0; c < channels; ++c) {
            output[base_idx + c * spatial_size] /= sum_exp;
        }
    }
}

// Comparision of two arrays
bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal 
}

int main() {
    //A random 4D tensor (e.g., [batch_size, channels, height, width])
    torch::Tensor input = torch::rand({1, 2,32,32});
    float* h_ptr = input.data_ptr<float>();

    int batch_size = 1;
    int channels = 2;
    int height = 32;
    int width = 32;
    size_t total_size = batch_size * channels * height * width;

    // Allocating memory on the device
    float *d_input, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_ptr, total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size * height * width + threads_per_block - 1) / threads_per_block;
    softmaxKernel<<<num_blocks, threads_per_block>>>(d_input, d_output, batch_size, channels, height, width);

    // Copy output to host
    float* cuda_output = new float[total_size];
    cudaMemcpy(cuda_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Apply PyTorch softmax for inferencing the output
    torch::Tensor output = torch::softmax(input, /*dim=*/1);

    //std::cout << "Softmax Tensor (PyTorch): " << output << std::endl;

    // Validating the CUDA output with PyTorch output
    float* h_output_ptr = output.data_ptr<float>();
    if (compareArrays(h_output_ptr, cuda_output, total_size, 1e-3))
        std::cout << "Test passed " << std::endl;
    else
        std::cout << "Test failed " << std::endl;

    // freeing up space
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] cuda_output;

    return 0;
}