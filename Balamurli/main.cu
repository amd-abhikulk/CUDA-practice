#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat colorImage = cv::imread("German-Shepherd-dog-Alsatian.jpg");
    
    // Check if the image was loaded successfully
    if (colorImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Create a Mat to hold the grayscale image
    cv::Mat grayImage;

    // Convert the color image to grayscale
    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);

    // Save the grayscale image
    cv::imwrite("./build/grayscale_image_cpu.jpg", grayImage);

    // Input image
    uchar* image_data = image.data;

    //Store your cuda_output in this pointer
    uchar* cuda_output;


    cv::Mat cudaOutput(grayImage);
    cudaOutput.data = cuda_output;
    cv::imwrite("./build/grayscale_image_gpu.jpg", cudaOutput);

    // Optionally, display the images

    // Wait for a key press
    cv::waitKey(0);

    return 0;
}
