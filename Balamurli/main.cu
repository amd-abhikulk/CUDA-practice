#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void convertImage( uchar* input, uchar* output, int height, int width)
{
    // Calculate thread indices for x and y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate the index for the pixel in the input and output arrays
        int idx = (y * width + x) * 3;  // BGR channels
        int gray_idx = y * width + x;    // Single channel for grayscale

        // Read RGB values from input
        uchar blue = input[idx]; //gives intensity of blue in that pixel, ranges from 0-255
        uchar green = input[idx + 1]; //gives the intensity of green 
        uchar red = input[idx + 2]; //gives the inensity of red
        
        // Convert to grayscale and store in output
        output[gray_idx] = static_cast<uchar>(
            0.299f * red +
            0.587f * green +
            0.114f * blue
        );
    }
}

int main() {
    // Load the image
    cv::Mat colorImage = cv::imread("/home/arunGPU12/bala/CUDAMasterclass/Assignment/CUDA-practice/Balamurli/German-Shepherd-dog-Alsatian.jpg");
    
    // Check if the image was loaded successfully
    if (colorImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Get image dimensions
    int height = colorImage.rows;
    int width = colorImage.cols;

    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 2D block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Allocate device memory
    uchar *d_input, *d_output;
    cudaMalloc(&d_input, height * width * 3 * sizeof(uchar));
    cudaMalloc(&d_output, height * width * sizeof(uchar));
    
    // Copy input image to device
    cudaMemcpy(d_input, colorImage.data, 
               height * width * 3 * sizeof(uchar),
               cudaMemcpyHostToDevice);

    // Launch kernel
    convertImage<<<gridSize, blockSize>>>(d_input, d_output, height, width);

    
    cv::Mat grayImage; 
    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::imwrite("./grayscale_image_cpu.jpg", grayImage);

    // Get results from GPU
    uchar* h_output = new uchar[height * width];
    cudaMemcpy(h_output, d_output, 
               height * width * sizeof(uchar),
               cudaMemcpyDeviceToHost);

   
    cv::Mat cudaOutput(height, width, CV_8UC1, h_output);
    cv::imwrite("./grayscale_image_gpu.jpg", cudaOutput);
    cv::waitKey(0);

    // free the memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}
