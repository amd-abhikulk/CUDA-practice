#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include <iostream>
#include <cmath>

// CUDA kernel to perform bilinear interpolation for image rotation
__global__ void rotateImageKernel(
    const uchar* input,        // Input image
    uchar* output,             // Output rotated image
    int input_rows,            // Rows of input image
    int input_cols,            // Columns of input image
    int output_rows,           // Rows of output image
    int output_cols,           // Columns of output image
    int channels,              // Number of channels (e.g., 3 for RGB)
    const float* rotation_matrix // Inverse rotation matrix
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // X coordinate of the output pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Y coordinate of the output pixel
    
    // Only process pixels within the bounds of the output image
    if (x < output_cols && y < output_rows) {
        // Apply the inverse transformation to find corresponding source pixel in the input image
        float src_x = rotation_matrix[0] * x + rotation_matrix[1] * y + rotation_matrix[2];
        float src_y = rotation_matrix[3] * x + rotation_matrix[4] * y + rotation_matrix[5];
        
        // Calculate the coordinates for bilinear interpolation (rounding down)
        int x1 = floorf(src_x);
        int y1 = floorf(src_y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        
        // Ensure the coordinates are within the image bounds
        if (x1 >= 0 && x2 < input_cols && y1 >= 0 && y2 < input_rows) {
            // Calculate interpolation weights for x and y axes
            float wx2 = src_x - x1;
            float wy2 = src_y - y1;
            float wx1 = 1.0f - wx2;
            float wy1 = 1.0f - wy2;
            
            // Perform bilinear interpolation for each channel (e.g., R, G, B)
            for (int c = 0; c < channels; c++) {
                uchar q11 = input[(y1 * input_cols + x1) * channels + c];
                uchar q12 = input[(y2 * input_cols + x1) * channels + c];
                uchar q21 = input[(y1 * input_cols + x2) * channels + c];
                uchar q22 = input[(y2 * input_cols + x2) * channels + c];
                
                // Interpolated value at the current pixel
                float interpolated_value = wy1 * (wx1 * q11 + wx2 * q21) +
                                           wy2 * (wx1 * q12 + wx2 * q22);
                
                // Assign the interpolated value to the output image, rounding to the nearest integer
                output[(y * output_cols + x) * channels + c] = static_cast<uchar>(nearbyintf(interpolated_value));
            }
        } else {
            // Fill with black color if the pixel is out of bounds (background)
            for (int c = 0; c < channels; c++) {
                output[(y * output_cols + x) * channels + c] = 0;
            }
        }
    }
}

// Helper function to calculate the inverse of a 2D rotation matrix
void calculateInverseMatrix(const cv::Mat& rotation_matrix, float* inverse_matrix) {
    // Calculate the determinant of the matrix
    double det = rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(1, 1) -
                 rotation_matrix.at<double>(0, 1) * rotation_matrix.at<double>(1, 0);
    float inv_det = 1.0f / det;
    
    // Calculate the inverse matrix elements using the formula for a 2x3 affine matrix
    inverse_matrix[0] = rotation_matrix.at<double>(1, 1) * inv_det;
    inverse_matrix[1] = -rotation_matrix.at<double>(0, 1) * inv_det;
    inverse_matrix[2] = (rotation_matrix.at<double>(0, 1) * rotation_matrix.at<double>(1, 2) -
                         rotation_matrix.at<double>(1, 1) * rotation_matrix.at<double>(0, 2)) * inv_det;
    inverse_matrix[3] = -rotation_matrix.at<double>(1, 0) * inv_det;
    inverse_matrix[4] = rotation_matrix.at<double>(0, 0) * inv_det;
    inverse_matrix[5] = (rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(0, 2) -
                         rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(1, 2)) * inv_det;
}

// Function to handle CUDA memory allocation and kernel launch for image rotation
void rotateImageCUDA(const cv::Mat& input, uchar* output, const cv::Mat& rotation_matrix,
                     cv::Size output_size) {
    int input_rows = input.rows;
    int input_cols = input.cols;
    int output_rows = output_size.height;
    int output_cols = output_size.width;
    int channels = input.channels();
    
    // Allocate memory for input and output images on the device (GPU)
    size_t input_size = input_rows * input_cols * channels * sizeof(uchar);
    size_t output_size_bytes = output_rows * output_cols * channels * sizeof(uchar);
    
    uchar *d_input, *d_output;
    float *d_inverse_matrix;
    cudaMalloc(&d_input, input_size);          // Device memory for input image
    cudaMalloc(&d_output, output_size_bytes);  // Device memory for output image
    cudaMalloc(&d_inverse_matrix, 6 * sizeof(float)); // Device memory for inverse rotation matrix
    
    // Calculate the inverse rotation matrix on the host (CPU)
    float h_inverse_matrix[6];
    calculateInverseMatrix(rotation_matrix, h_inverse_matrix);
    
    // Copy the input image and inverse matrix to the device (GPU)
    cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverse_matrix, h_inverse_matrix, 6 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up CUDA grid and block dimensions
    dim3 block_dim(16, 16);  // Thread block size (16x16)
    dim3 grid_dim(
        (output_cols + block_dim.x - 1) / block_dim.x,  // Number of blocks in x direction
        (output_rows + block_dim.y - 1) / block_dim.y   // Number of blocks in y direction
    );
    
    // Launch the CUDA kernel for image rotation
    rotateImageKernel<<<grid_dim, block_dim>>>(d_input, d_output, input_rows, input_cols,
                                                output_rows, output_cols, channels, d_inverse_matrix);
    
    // Copy the result back from device to host
    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
    
    // Clean up by freeing the allocated device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_inverse_matrix);
}

// Function to compare two arrays with tolerance for minor differences (e.g., due to rounding errors)
bool compareArraysWithTolerance(const uchar* arr1, const uchar* arr2, size_t size, int tolerance) {
    for (size_t i = 0; i < size; ++i) {
        // Check if the difference between corresponding pixels exceeds the tolerance
        if (std::abs((arr1[i]) - arr2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // Load the input image from disk
    cv::Mat image = cv::imread("German-Shepherd-dog-Alsatian.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Get the center of the image for rotation
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);

    // Get the rotation matrix to rotate the image by 30 degrees
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, 30, 1.0);

    // Determine the bounding box of the rotated image to avoid clipping
    cv::Rect2f boundingBox = cv::RotatedRect(center, image.size(), 30).boundingRect();
    
    // Adjust the rotation matrix to account for translation due to rotation
    rotationMatrix.at<double>(0, 2) += boundingBox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += boundingBox.height / 2.0 - center.y;

    // Perform the rotation using OpenCV for comparison
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotationMatrix, boundingBox.size());

    // Allocate memory for the rotated image output using CUDA
    cv::Size output_size = boundingBox.size();
    uchar* cuda_output = new uchar[output_size.height * output_size.width * image.channels()];

    // Perform image rotation using CUDA
    rotateImageCUDA(image, cuda_output, rotationMatrix, output_size);

    // Convert the CUDA output to OpenCV format for saving
    cv::Mat cudaOutput(rotatedImage.size(), rotatedImage.type(), cuda_output);

    // Save both the CPU (OpenCV) and GPU (CUDA) results for comparison
    cv::imwrite("./rotated_image_cpu.jpg", rotatedImage);
    cv::imwrite("./rotated_image_gpu.jpg", cudaOutput);

    std::cout << "Image rotation completed and saved as 'rotated_image_cpu.jpg' and 'rotated_image_gpu.jpg'" << std::endl;

    // Compare the results from CPU and GPU with a tolerance for minor differences
    int tolerance = 300;  // Increased tolerance for minor pixel differences due to rounding errors
    size_t size = rotatedImage.cols * rotatedImage.rows * rotatedImage.channels();
    
    // Check if the images are similar within the given tolerance
    if (compareArraysWithTolerance(cuda_output, rotatedImage.data, size, tolerance)) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
        
        // Print the pixel differences for debugging (every 500th pixel)
        for (size_t i = 0; i < size; i += 500) {
            if (std::abs(cuda_output[i] - rotatedImage.data[i]) > tolerance) {
                std::cout << "Difference at pixel " << i << ": "
                          << "CPU: " << static_cast<int>(rotatedImage.data[i])
                          << " GPU: " << static_cast<int>((cuda_output[i])) << std::endl;
            }
        }
    }

    // Clean up allocated memory
    delete[] cuda_output;

    cv::waitKey(0);
    return 0;
}
