#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include <iostream>

bool compareArrays(const uchar* arr1, const uchar* arr2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return false; // Arrays are not equal
        }
    }
    return true; // Arrays are equal within the specified tolerance
}


/*
Write your kernel code here
*/

int main() {
    // Load the image
    cv::Mat image = cv::imread("German-Shepherd-dog-Alsatian.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Get the image center
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);

    // Get the rotation matrix for rotating the image by 30 degrees
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, 30, 1.0);
    // Input image
    uchar* image_data = image.data;


    // Determine the bounding box of the rotated image
    cv::Rect2f boundingBox = cv::RotatedRect(center, image.size(), 30).boundingRect();

    // Adjust the rotation matrix to take into account the translation
    rotationMatrix.at<double>(0, 2) += boundingBox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += boundingBox.height / 2.0 - center.y;

    // Roatation matrix data
    uchar* rotMat = rotationMatrix.data;
    // Rotate the image
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotationMatrix, boundingBox.size());

    //Store your cuda_output in this input
    uchar* cuda_output;
    uchar* host_output = rotatedImage.data;
    // Display the original and rotated images
    // cv::imshow("Original Image", image);
    // cv::imshow("Rotated Image", rotatedImage);
    cv::Mat cudaOutput(rotatedImage);
    cudaOutput.data = cuda_output;
    cv::imwrite("./build/rotated_image_cpu.jpg", rotatedImage);
    cv::imwrite("./build/rotated_image_gpu.jpg", cudaOutput);

    size_t size = rotatedImage.cols * rotatedImage.rows * rotatedImage.channels();
    if(compareArrays(cuda_output, host_output, size))
        std::cout << "Test passed" << std::endl;
    else
        std::cout << "Test failed" << std::endl;


    // Wait for a key press and exit
    cv::waitKey(0);
    return 0;
}
