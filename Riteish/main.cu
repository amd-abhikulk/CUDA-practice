#include <iostream>
#include <torch/torch.h>
#include "cuda_runtime.h"

using namespace std;

bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal within the specified tolerance
}


/*
Write your kernel code here
*/
__global__ void average_pooling(
    const float* input, float* output,
    int channels, 
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride)
{
    int c = blockIdx.z;  // Channel index
    int h = blockIdx.y * blockDim.y + threadIdx.y; // Height index of the output
    int w = blockIdx.x * blockDim.x + threadIdx.x; // Width index of the output

    // Applying average pooling on CUDA
    if (c < channels && h < output_height && w < output_width) {
        float sum = 0.0f;
        int input_start_h = h * stride; // Starting height for input
        int input_start_w = w * stride; // Starting width for input

        // Calculating sum within kernel
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_h = input_start_h + i;
                int input_w = input_start_w + j;

                // Ensure we are within bounds before adding to the sum
                if (input_h < input_height && input_w < input_width) {
                    sum += input[(c * input_height + input_h) * input_width + input_w];
                }
            }
        }

        // Calculate average and write to output
        output[(c * output_height + h) * output_width + w] = sum / (kernel_size * kernel_size);
    }
}




int main()
{
    // torch::Tensor input = torch::rand({1, 3, 1024, 1024}, torch::dtype(torch::kFloat)); // Random tensor


    // input sizes and other parameters
    const int batch_size = 1;
    const int channels = 3;
    const int input_height = 1024;
    const int input_width = 1024;
    const int kernel_size = 2;
    const int stride = 2;

    // Calculating output dimensions
    const int output_height = (input_height - kernel_size) / stride + 1;
    const int output_width = (input_width - kernel_size) / stride + 1;

    // Allocating host memory for input and output
    torch::Tensor input = torch::rand({batch_size, channels, input_height, input_width}, torch::dtype(torch::kFloat));

    std::cout << "Input Tensor Size: " << input.sizes() << std::endl;

    // // Define the kernel size and stride
    // std::vector<int64_t> kernel_size = {2, 2};
    // std::vector<int64_t> stride = {2, 2};

    //Host input
    float* h_ptr = input.data_ptr<float>();

    // // input tensor
    // std::cout << "Input Tensor: " << input << std::endl;

    // Allocating memory on the device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * batch_size * channels * input_height * input_width);
    cudaMalloc((void**)&d_output, sizeof(float) * batch_size * channels * output_height * output_width);

    // Copying input from host to device
    cudaMemcpy(d_input, h_ptr, sizeof(float) * batch_size * channels * input_height * input_width, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid((output_width+31)/32, (output_height+31)/32, 3);
    dim3 block(32, 32);

    // the average pooling kernel
    average_pooling<<<grid, block>>>(d_input, d_output, channels, input_height, input_width, output_height, output_width, kernel_size, stride);

    // Copying result back to host
    float* cuda_output = (float*)malloc(sizeof(float) * batch_size * channels * output_height * output_width);
    cudaMemcpy(cuda_output, d_output, sizeof(float) * batch_size * channels * output_height * output_width, cudaMemcpyDeviceToHost);

    // // CUDA output
    // std::cout << "CUDA Output: ";
    // for (int i = 0; i < batch_size * channels * output_height * output_width; ++i) {
    //     std::cout << cuda_output[i] << " ";
    // }
    // std::cout << std::endl;

    // Comparing with the PyTorch output
    torch::nn::AvgPool2d pool(torch::nn::AvgPool2dOptions(kernel_size).stride(stride));
    torch::Tensor output = pool(input);

    // // PyTorch output
    // std::cout << "PyTorch Output: " << output << std::endl;

    // Comparing outputs
    size_t output_size = 1;
    for (int i : output.sizes())
        output_size *= i;

    float* h_output_ptr = output.data_ptr<float>();
    if (compareArrays(cuda_output, h_output_ptr, output_size, 0.01))
        std::cout << "Test passed" << std::endl;
    else
        std::cout << "Test failed" << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(cuda_output);

    return 0;
}
