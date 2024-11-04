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


#define CUDA_CHECK(call)                                     \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            fprintf(stderr, "CUDA error in %s (%d): %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err);                                       \
        }                                                    \
    } while(0)

/*
Write your kernel code here
*/

__global__ void maxPoolOP(float *in, float *out, int size, int d_size){
    int tid_x = threadIdx.x;
    
    int row = blockIdx.x * 2;
    int col = threadIdx.x * 2;

    //Each thread will serach the kerenl mapped cells of the image
    float mx1 = fmaxf(fmaxf(in[row*size + col],in[row*size + col + 1]),in[row*size + col + 2]);
    float mx2 = fmaxf(fmaxf(in[row*size + size + col],in[row*size + size + col + 1]),in[row*size + size + col + 2]);
    float mx3 = fmaxf(fmaxf(in[row*size + 2*size + col],in[row*size + 2*size + col + 1]),in[row*size + 2*size + col + 2]);

    // The max among the 3x3 i.e total 9 cells is stored in the output image
    out[blockIdx.x*d_size + tid_x] = fmaxf(mx1, fmaxf(mx2,mx3));

    //printf("%d : %f\n", blockIdx.x*d_size + tid_x ,out[blockIdx.x*d_size + tid_x]);

}

int main()
{
    torch::Tensor input = torch::rand({1, 3, 1024, 1024}, torch::dtype(torch::kFloat)); // Random tensor
    std::cout << "Input Tensor Size: " << input.sizes() << std::endl;

    // Define the kernel size and stride
    std::vector<int64_t> kernel_size = {3, 3};     //changed
    std::vector<int64_t> stride = {2, 2};

    //Host input - tensor is converted to flattend array
    float* h_ptr = input.data_ptr<float>();

    // Apply average pooling
    torch::nn::MaxPool2d pool(torch::nn::MaxPool2dOptions(3).stride(2)); 

    // Apply the pooling operation
    torch::Tensor output = pool(input);

    cout << output.sizes()<<endl;

    size_t output_size =1;
    for(int i : output.sizes())
        output_size *= i;
    
    float* h_output_ptr = output.data_ptr<float>();

    int s = output.numel(); // Get the number of elements in the tensor
    
    //_____________________________________________________________________
    
    //Based on the input dimensions, getting the output dimension
    int input_size = input.size(2);   //Assumming that h-dim and w-dim is same
    int down_sample_size = ((input_size - kernel_size[0])/2) + 1;

    //cout << "down_sample_size : " << down_sample_size << endl;
    
    // Creating seperate arrays which stores each channel's input and output(2d array)
    float* h_a = (float*)malloc(input_size * input_size * sizeof(float));
    float* h_b = (float*)malloc(input_size * input_size * sizeof(float));
    float* h_c = (float*)malloc(input_size * input_size * sizeof(float));
    float* h_a_out = (float*)malloc(down_sample_size * down_sample_size * sizeof(float));
    float* h_b_out = (float*)malloc(down_sample_size * down_sample_size * sizeof(float));
    float* h_c_out = (float*)malloc(down_sample_size * down_sample_size * sizeof(float));
    
    //Copy raw_data from the h_ptr to the separate arrays
    //so that maxpool operration will work seperately on each channel
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            h_a[i * input_size + j] = h_ptr[i * input_size + j];
            h_b[i * input_size + j] = h_ptr[1 * (input_size * input_size) + i * input_size + j];
            h_c[i * input_size + j] = h_ptr[2 * (input_size * input_size) + i * input_size + j];
        }
    }

    //Allocating memory on device for arrays to stores each channel(2d array)
    float *d_a, *d_b, *d_c, *d_a_out, *d_b_out, *d_c_out;
    cudaMalloc((void**)&d_a, input_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_b, input_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_c, input_size * input_size * sizeof(float));
    cudaMalloc((void**)&d_a_out, down_sample_size * down_sample_size * sizeof(float));
    cudaMalloc((void**)&d_b_out, down_sample_size * down_sample_size * sizeof(float));
    cudaMalloc((void**)&d_c_out, down_sample_size * down_sample_size * sizeof(float));

    // Copying the array on the device
    cudaMemcpy(d_a, h_a, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    //Calling the kernel to work on each channel
    //dim3 block_size(down_sample_size,down_sample_size);

    //Using Cuda streams to asynchronously run the kerenls
    cudaStream_t str1, str2, str3;
    
    cudaStreamCreate(&str1);
    cudaStreamCreate(&str2);
    cudaStreamCreate(&str3);

    maxPoolOP<<<down_sample_size,down_sample_size, 0, str1>>> (d_a,d_a_out,input_size,down_sample_size);
    maxPoolOP<<<down_sample_size,down_sample_size, 0, str2>>> (d_b,d_b_out,input_size,down_sample_size);
    maxPoolOP<<<down_sample_size,down_sample_size, 0, str3>>> (d_c,d_c_out,input_size,down_sample_size);

    cudaError_t error = cudaGetLastError();
    CUDA_CHECK(error);
    
    cudaDeviceSynchronize();

    //copying the output calculated, back from device to host 
    cudaMemcpy(h_a_out, d_a_out, down_sample_size * down_sample_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out, d_b_out, down_sample_size * down_sample_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_out, d_c_out, down_sample_size * down_sample_size * sizeof(float), cudaMemcpyDeviceToHost);

    //Combing the resultant arrays in one
    float *cuda_output = (float*)malloc(3 * down_sample_size * down_sample_size * sizeof(float));
    
    int size = down_sample_size * down_sample_size;
    
    memcpy(cuda_output, h_a_out, size * sizeof(float));
    memcpy(cuda_output + size, h_b_out, size * sizeof(float));
    memcpy(cuda_output + 2 * size, h_c_out, size * sizeof(float));

    // Printing the downsampled image
    // cout <<"___________________________"<<endl;
    // for(int i = 0; i < 3*down_sample_size*down_sample_size; i++){
    //     cout << i << " : " <<cuda_output[i] << endl;
    // }

    //_____________________________________________________________________________________________________

    if(compareArrays(h_output_ptr, cuda_output, output_size, 1e-3))
        std::cout << "Test passed " << std::endl;
    else
        std::cout << "Test failed " << std::endl;

    cudaStreamDestroy(str1);
    cudaStreamDestroy(str2);
    cudaStreamDestroy(str3);
        
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_a_out);
    cudaFree(d_b_out);
    cudaFree(d_c_out);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a_out);
    free(h_b_out);
    free(h_c_out);
    free(cuda_output);
    
    return 0;

}