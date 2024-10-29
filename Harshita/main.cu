#include <torch/torch.h>
#include <iostream>
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

/*Write your kernel code here*/


int main() {
    // Create a random 4D tensor (e.g., [batch_size, channels, height, width])
    torch::Tensor input = torch::rand({1, 8, 1024, 1024}); 

    float* h_ptr = input.data_ptr<float>();

    //Store your cuda_output in this input
    float* cuda_output;
    // Apply softmax on the specified dimension (e.g., channels)
    // Here, we apply softmax along dimension 1 (the channels)
    torch::Tensor output = torch::softmax(input, /*dim=*/1);

    // Print the softmax output
    std::cout << "Softmax Tensor: " << output << std::endl;
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
