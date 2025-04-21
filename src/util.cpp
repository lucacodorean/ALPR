//
// Created by codor on 4/6/2025.
//

#include "util.h"
#include <cmath>
#include <opencv2/highgui.hpp>

using namespace std;

vector<vector<double>> ALPR::Util::generateGaussianKernel(int kernelSize, double sigma) {
    vector kernel(kernelSize, vector<double>(kernelSize));
    double sum = 0.0;

    int half = kernelSize / 2;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            double value = exp(-(i * i + j * j) / (2.0 * sigma * sigma)) / (2.0 * PI * sigma * sigma);
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

int* ALPR::Util::computeHistogram(cv::Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    int* histogram = (int*)calloc(256, sizeof(int));

    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++) histogram[source.at<uchar>(i,j)]++;
    }

    return histogram;
}

bool ALPR::Util::isBlue(const cv::Vec3b& pixel) {
    int H = pixel[0];
    int S = pixel[1];
    int V = pixel[2];
    return (H >= 100 && H <= 130 && S >= 150 && V >= 50);
}
