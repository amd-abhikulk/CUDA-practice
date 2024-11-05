#include <torch/torch.h>
#include <iostream>


bool compareArrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > tolerance) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal within the specified tolerance
}

/* Write your kernel code here*/

int main() {
    // Initialize the tensor with random values for demonstration
    torch::manual_seed(42);  // Set seed for reproducibility
    torch::Tensor input = torch::randn({1024, 1024, 32});  // Random tensor with shape (1024, 1024, 32)

    std::cout << "Original Tensor Shape: " << input.sizes() << std::endl;


    //Host input - tensor is converted to flattend array
    float* h_ptr = input.data_ptr<float>();
    // Compute the RMS normalization
    // Step 1: Compute square of the tensor
    auto tensor_squared = input.pow(2);
    
    // Step 2: Compute the mean along all dimensions (axis 0, 1, and 2)
    auto mean_squared = tensor_squared.mean();  // Scalar value, mean of all elements
    
    // Step 3: Compute the RMS
    auto rms = mean_squared.sqrt();  // Square root of the mean squared value
    
    // Step 4: Normalize the tensor
    torch::Tensor output = input / rms;

    // Use the following variable for output of your kernel.
    float *cuda_output;

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
