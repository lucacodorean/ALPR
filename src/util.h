//
// Created by codor on 4/6/2025.
//

#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/version.hpp>

#ifndef CONSTANTS_
    #define CONSTANTS_

    #define BLUE_CHANNEL    0
    #define RED_CHANNEL     1
    #define GREEN_CHANNEL   2

    #define KERNEL_SIZE     5
    #define SIGMA_VALUE     0.75
    #define BW_THRESHOLD    150

    #define PI              3.14159265358979323846
    #define SUCCESS         0
    #define FAILURE         1

#endif


using namespace std;

namespace ALPR {
    class Util {
        public:

        /**
         * This function will try so simulate the kernel that cv::GaussianBlur uses.
         * The kernel can be seen as a window that moves across the matrix.
         * The implementation follows the DP methodology.
         * @param kernelSize should be an odd value. Describes the dimensions of the matrix
         * @param sigma      is the standard deviation. Based on the value, the kernel is affected.
         * Should be a value between 0 and 1 in order to keep the effect applied by the kernel sharp and narrow.
         * @return A matrix of double elements that is normalized.
         */
        static vector<vector<double>> generateGaussianKernel(int kernelSize, double sigma);
        static int*  computeHistogram(cv::Mat);

    };
}

#endif //UTIL_H
