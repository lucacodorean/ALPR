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

int* ALPR::Util::computeHistogram(Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    int* histogram = (int*)calloc(256, sizeof(int));

    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++) histogram[source.at<uchar>(i,j)]++;
    }

    return histogram;
}

bool ALPR::Util::isBlue(const Vec3b& pixel) {
    int H = pixel[0];
    int S = pixel[1];
    int V = pixel[2];
    return H >= 100 && H <= 130 && S >= 150 && V >= 50;
}

bool ALPR::Util::isInside(int img_rows, int img_cols, int i, int j) {
    return i<img_rows && i>=0 && j<img_cols && j>=0;
}

Mat ALPR::Util::erosion(Mat source, neighborhood_structure neighborhood, int no_iter){
    Mat dst, aux;

    dst = Mat(source.rows, source.cols, source.type(), Scalar::all(255));
    aux = source.clone();

    for (int current_iteration = 0; current_iteration<no_iter; current_iteration++) {

        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                if (aux.at<uchar>(i,j) == 255) continue;

                for (int k = 0; k < neighborhood.size; k++) {
                    int next_i = i + neighborhood.di[k];
                    int next_j = j + neighborhood.dj[k];

                    if (isInside(source.rows, source.cols, next_i, next_j))
                        dst.at<uchar>(next_i, next_j) = 0;
                }
            }
        }

        std::swap(aux, dst);
    }

    // dst = aux.clone(); //AICI DADEAM CLONE LA AUXILIAR INCA O DATA
    return dst;

}

Mat ALPR::Util::dilation(Mat source, neighborhood_structure neighborhood, int no_iter){
    Mat dst, aux;
    aux = source.clone();

    for (int current_iteration = 0; current_iteration<no_iter; current_iteration++) {
        dst = aux.clone();

        for (int i = 0; i<source.rows; i++) {
            for (int j = 0; j<source.cols; j++) {
                bool ok = true;
                for (int k = 0; k<neighborhood.size && ok; k++) {
                    int next_i = i + neighborhood.di[k];
                    int next_j = j + neighborhood.dj[k];
                    if (isInside(dst.rows, dst.cols, next_i, next_j) && aux.at<uchar>(next_i, next_j) == 255) ok = false;
                }

                dst.at<uchar>(i, j) = ok ? 0 : 255;
            }
        }
        std::swap(aux, dst);
    }

    dst = aux.clone();
    return dst;
}

Mat ALPR::Util::closing(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();
    Mat dst;

    for (int i = 0; i < no_iter; ++i) {
        dst = aux.clone();
        aux = dilation(dst, neighborhood, 1);
        dst = erosion(aux, neighborhood, 1);
        aux = dst.clone();
    }

    return aux;
}

Mat ALPR::Util::opening(Mat source, neighborhood_structure neighborhood, int no_iter) {

    Mat dst, aux;
    aux = source.clone();

    for (int i = 0 ; i<no_iter; i++) {
        dst = aux.clone();
        aux = erosion(dst, neighborhood, 1);
        dst = dilation(aux, neighborhood, 1);
        aux = dst.clone();
    }

    return dst;
}