//
// Created by codor on 4/6/2025.
//

#include "util.h"
#include <queue>
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

    return (H >= 102 && H <= 118) && (S >= 160 && S <= 255) && (V >= 70  && V <= 200);
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

std::pair<int, int> applyKernel(Mat source, int x, int y, std::vector<std::vector<int>> sobelX, std::vector<std::vector<int>> sobelY) {
    int gx = 0, gy = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j<=1; j++) {
            int pixel = source.at<uchar>(x+i, y+j);
            gx += sobelX[i+1][j+1] * pixel;
            gy += sobelY[i+1][j+1] * pixel;
        }
    }

    return {gx, gy};
}

int getModule(int first, int second) {
    return sqrt(first*first + second*second);
}

Gradients ALPR::Util::retrieveGradients(Mat source, std::vector<std::vector<int>> sobelX, std::vector<std::vector<int>> sobelY)  {
    Mat result      = Mat(source.rows, source.cols, CV_32F, Scalar(0));
    Mat gradientsX  = Mat(source.rows, source.cols, CV_32F, Scalar(0));
    Mat gradientsY  = Mat(source.rows, source.cols, CV_32F, Scalar(0));

    for (int i = 1; i<source.rows -1; i++) {
        for (int j = 1; j<source.cols -1; j++) {
            std::pair<int, int> gradient = applyKernel(source, i, j, sobelX, sobelY);
            int module =  getModule(gradient.first, gradient.second);

            if (module > 255) module = 255;
            result.at<float>(i, j) = module;
            gradientsX.at<float>(i, j) = gradient.first;
            gradientsY.at<float>(i, j) = gradient.second;
        }
    }

    return {gradientsX, gradientsX, result};
}

Mat ALPR::Util::retrieveSuppression(Gradients gradients) {
    Mat result = Mat::zeros(gradients.module.rows, gradients.module.cols, CV_8UC1);

    for (int i = 1; i < gradients.module.rows -1; i++) {
        for (int j = 1; j<gradients.module.cols -1; j++) {
            float angle = (float)atan2(gradients.gy.at<float>(i,j), gradients.gx.at<float>(i,j)) / PI;
            if (angle < 0) angle += 100;

            float mag = gradients.module.at<float>(i, j);
            float mag1 = 0, mag2 = 0;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                mag1 = gradients.module.at<float>(i, j - 1);
                mag2 = gradients.module.at<float>(i, j + 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                mag1 = gradients.module.at<float>(i - 1, j + 1);
                mag2 = gradients.module.at<float>(i + 1, j - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                mag1 = gradients.module.at<float>(i - 1, j);
                mag2 = gradients.module.at<float>(i + 1, j);
            } else if (angle >= 112.5 && angle < 157.5) {
                mag1 = gradients.module.at<float>(i - 1, j - 1);
                mag2 = gradients.module.at<float>(i + 1, j + 1);
            }

            if (mag >= mag1 && mag >= mag2) result.at<uchar>(i, j) = (int)mag;
        }
    }

    return result;
}

Mat ALPR::Util::applyHysteresis(Mat source, int lowThreshold, int highThreshold) {
    Mat result = Mat::zeros(source.rows, source.cols, source.type());

    std::queue<pair<int,int>> queue;
    for (int i = 0; i<source.rows; i++) {
        for (int j = 0; j<source.cols; j++) {
            if (source.at<uchar>(i, j) > highThreshold) {
                result.at<uchar>(i, j) = STRONG_EDGE;
                queue.push({i, j});
            }
            else if (source.at<uchar>(i,j) >= lowThreshold) result.at<uchar>(i, j) = WEAK_EDGE;
            else result.at<uchar>(i, j) = NO_EDGE;
        }
    }

    while (!queue.empty()) {
        pair<int,int> front = queue.front();
        queue.pop();

        for (int k = 0; k<neighborhood.size; k++) {
            int next_i = front.first + neighborhood.di[k];
            int next_j = front.second + neighborhood.dj[k];

            if (isInside(source.rows, source.cols, next_i, next_j) && result.at<uchar>(next_i, next_j) == WEAK_EDGE) {
                result.at<uchar>(next_i, next_j) = STRONG_EDGE;
                queue.push({next_i, next_j});
            }
        }
    }

    for (int i = 0; i<source.rows; i++) {
        for (int j = 0; j<source.cols; j++) {
            if (result.at<uchar>(i,j) != STRONG_EDGE) result.at<uchar>(i,j) = NO_EDGE;
        }
    }

    return result;
}

void ALPR::Util::Canny(Mat source, Mat& output, int lowThreshold, int highThreshold) {
    Gradients gradients = retrieveGradients(source, getSobelKernelX(), getSobelKernelY());
    Mat temp =  retrieveSuppression(gradients);
    output =    applyHysteresis(temp, lowThreshold, highThreshold);
}