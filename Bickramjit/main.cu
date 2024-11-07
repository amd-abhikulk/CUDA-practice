#include <iostream>
#include <torch/torch.h>
#include "cuda_runtime.h"
#include <cmath>

using namespace std;

// Function to compare arrays
bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal within the specified tolerance
}

// CUDA kernel for convolution
__global__ void ImageConvolution(const float* input, float* output, const float* weights, 
                                 int input_channels, int output_channels,
                                 int input_height, int input_width,
                                 int output_height, int output_width,
                                 int filter_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int output_channel = blockIdx.z;

    if (idx < output_width && idy < output_height) {
        // Initialize output element
        float sum = 0.0f;

        // Apply the filter for the output channel
        for (int c = 0; c < input_channels; ++c) { // Loop over input channels
            for (int fx = 0; fx < filter_size; ++fx) {  // Loop over filter width
                for (int fy = 0; fy < filter_size; ++fy) {  // Loop over filter height
                    int input_x = idx * stride + fx;
                    int input_y = idy * stride + fy;
                    int input_index = (c * input_height + input_y) * input_width + input_x;
                    int weight_index = ((output_channel * input_channels + c) * filter_size + fy) * filter_size + fx;
                    sum += input[input_index] * weights[weight_index];
                }
            }
        }

        // Store the result in the output array
        int output_index = (output_channel * output_height + idy) * output_width + idx;
        output[output_index] = sum;
    }
}

int main() {
    // Define input tensor
    torch::Tensor input = torch::rand({1, 3, 1024, 1024}, torch::dtype(torch::kFloat)); // Random tensor
    std::cout << "Input Tensor Size: " << input.sizes() << std::endl;

    // Host input pointer
    float* h_input = input.data_ptr<float>();

    // Define convolution layer and get output
    torch::nn::Conv2d convLayer(torch::nn::Conv2dOptions(3, 8, 3).stride(1).bias(false));
    torch::Tensor output = convLayer(input);
    float* h_output_ptr = output.data_ptr<float>();

    // Get weights and size for kernel
    float* h_weights = convLayer->weight.data_ptr<float>();
    int input_height = input.size(2);
    int input_width = input.size(3);
    int output_channels = output.size(1);
    int input_channels = input.size(1);
    int output_height = output.size(2);
    int output_width = output.size(3);
    int filter_size = 3;
    int stride = 1;

    // Allocate memory on the GPU
    float* d_input;
    float* d_output;
    float* d_weights;
    size_t input_size = input_channels * input_height * input_width * sizeof(float);
    size_t output_size = output_channels * output_height * output_width * sizeof(float);
    size_t weight_size = output_channels * input_channels * filter_size * filter_size * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_weights, weight_size);

    // Copy data to the GPU
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weight_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       output_channels);
    ImageConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_weights, 
                                                         input_channels, output_channels,
                                                         input_height, input_width,
                                                         output_height, output_width,
                                                         filter_size, stride);

    // Copy result back to host
    float* cuda_output = new float[output_channels * output_height * output_width];
    cudaMemcpy(cuda_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Compare results
    if (compareArrays(h_output_ptr, cuda_output, output_channels * output_height * output_width, 1e-3))
        std::cout << "Test passed " << std::endl;
    else
        std::cout << "Test failed " << std::endl;

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    delete[] cuda_output;

    return 0;
}

