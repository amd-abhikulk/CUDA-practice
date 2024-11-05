#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono> 

using namespace std;

// CUDA kernel for matrix multiplication with bias addition
__global__ void linear_layer_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int input_size, int output_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Index into batch (rows)
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Index into output (columns)

    if (row < batch_size && col < output_size) {
        // Perform dot product between input and weight row
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        // Add bias and store the result
        output[row * output_size + col] = sum + bias[col];
    }
}

// Compare arrays for equality within a tolerance
bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal within the specified tolerance
}

int main() {
    // Create a linear layer with 1024 input features and 128 output features
    torch::nn::Linear linear(1024, 128);

    // Create an input tensor with batch size 32 and input size 1024
    torch::Tensor input = torch::randn({32, 1024}); 

    // Timing CPU (PyTorch) execution
    auto cpu_start = chrono::high_resolution_clock::now();
    
    // Forward pass through the linear layer using PyTorch (for comparison)
    torch::Tensor output = linear(input);

    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    cout << "CPU (PyTorch) execution time: " << cpu_duration.count() << " ms" << endl;

    // Extract raw pointers from the input, weights, and bias
    float* h_input = input.data_ptr<float>();
    float *h_weight_ptr, *h_bias_ptr;
    for (const auto& pair : linear->named_parameters()) {
        std::string name = pair.key();
        torch::Tensor param = pair.value();

        if (name == "weight") {
            h_weight_ptr = param.data_ptr<float>();
        } else if (name == "bias") {
            h_bias_ptr = param.data_ptr<float>();
        }
    }

    // Allocate CUDA memory
    float *d_input, *d_weight, *d_bias, *d_output;
    size_t input_size = 32 * 1024 * sizeof(float);
    size_t weight_size = 128 * 1024 * sizeof(float);
    size_t bias_size = 128 * sizeof(float);
    size_t output_size = 32 * 128 * sizeof(float);

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weight, weight_size);
    cudaMalloc((void**)&d_bias, bias_size);
    cudaMalloc((void**)&d_output, output_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight_ptr, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias_ptr, bias_size, cudaMemcpyHostToDevice);

    // Configure CUDA kernel
    dim3 blockDim(16, 16); // Define block size (16x16 threads)
    dim3 gridDim((32 + blockDim.x - 1) / blockDim.x, (128 + blockDim.y - 1) / blockDim.y); // Grid size

    // Timing GPU execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Start recording GPU time

    // Launch kernel using CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    linear_layer_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_weight, d_bias, d_output, 32, 1024, 128);
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop); // Stop recording GPU time
    cudaEventSynchronize(stop);

    // Calculate the time elapsed on GPU
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    cout << "GPU (CUDA) execution time: " << gpu_time << " ms" << endl;

    // Allocate memory for the CUDA output on host
    float* h_cuda_output = new float[32 * 128];

    // Copy the CUDA output back to the host
    cudaMemcpy(h_cuda_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Compare the results with the PyTorch output
    float* h_output_ptr = output.data_ptr<float>();
    if (compareArrays(h_output_ptr, h_cuda_output, 32 * 128, 1e-3))
        std::cout << "Test passed " << std::endl;
    else
        std::cout << "Test failed " << std::endl;

    // Free CUDA memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    // Free host memory
    delete[] h_cuda_output;

    return 0;
}
