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


int main()
{
    torch::Tensor input = torch::rand({1, 3, 1024, 1024}, torch::dtype(torch::kFloat)); // Random tensor

    std::cout << "Input Tensor Size: " << input.sizes() << std::endl;

    //Host input
    float* h_ptr = input.data_ptr<float>();

    //Store your cuda_output in this input
    float* cuda_output;
    // Apply average pooling
    torch::nn::Conv2d convLayer(torch::nn::Conv2dOptions(3, 8, 3).stride(1).bias(false));

    // Apply the pooling operation
    torch::Tensor output = convLayer(input);
    float* weight = convLayer->weight.data_ptr<float>();


    // Print the output shape
    size_t output_size =1;
    for(int i : output.sizes())
        output_size *= i;

    float* h_output_ptr = output.data_ptr<float>();
    if(compareArrays(h_output_ptr, cuda_output, output_size, 1e-3))
        std::cout << "Test passed " << std::endl;
    else
        std::cout << "Test failed " << std::endl;
    
    return 0;

}