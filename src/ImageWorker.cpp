//
// Created by codor on 4/6/2025.
//

#include "ImageWorker.h"

#include "util.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


ALPR::ImageWorker::ImageWorker(const string& path) {
    if (path.empty()) {
        std::cout<<"Can't operate with this path.";
        return;
    }

    this->m_image = cv::imread(path, cv::IMREAD_COLOR);
}

int ALPR::ImageWorker::convertToGreyScale() {

    if (this->m_image.empty()) {
        std::cout<<"Image not set-up yet."<<std::endl;
        return FAILURE;
    }

    this->m_greyscaleImage = cv::Mat::zeros(this->m_image.rows, this->m_image.cols, CV_8UC1);

    for (int i = 0; i<this->m_image.rows; i++) {
        for (int j = 0; j<this->m_image.cols; j++) {
            this->m_greyscaleImage.at<uchar>(i, j) =
                (this->m_image.at<cv::Vec3b>(i, j)[BLUE_CHANNEL] + this->m_image.at<cv::Vec3b>(i, j)[GREEN_CHANNEL] + this->m_image.at<cv::Vec3b>(i, j)[RED_CHANNEL])/3 ;
        }
    }

    return SUCCESS;
}

int ALPR::ImageWorker::applyBlur(vector<vector<double>> kernel) {
    if (this->m_greyscaleImage.empty()) {
        std::cout << "Greyscale image not set-up yet." << std::endl;
        return FAILURE;
    }

    int kernelSize = kernel.size();
    int half = kernelSize / 2;
    
    this->m_blurredImage = this->m_greyscaleImage.clone();

    for (int y = half; y < this->m_greyscaleImage.rows - half; y++) {
        for (int x = half; x < this->m_greyscaleImage.cols - half; x++) {
            double sum = 0.0;

            for (int i = -half; i <= half; i++) {
                for (int j = -half; j <= half; j++) {
                    sum += this->m_greyscaleImage.at<uchar>(y + i, x + j) * 
                           kernel[i + half][j + half];
                }
            }
            
            this->m_blurredImage.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }

    return SUCCESS;
}

int getAdaptiveThreshold(Mat image) {
    int threshold = 0;
    int total = image.rows * image.cols;
    int* histogram = ALPR::Util::computeHistogram(image);

    float sum = 0;
    for (int i = 0; i < 256; i++) sum += i * histogram[i];

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float maxVariance = 0;
    threshold = 0;


    for (int i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB == 0) continue;

        wF = total - wB;
        if (wF == 0) break;

        sumB += i * histogram[i];
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;

        float variance = wB * wF * (mB - mF) * (mB - mF);

        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

int ALPR::ImageWorker::convertToBinary() {
    if (this->m_blurredImage.type() != CV_8UC1) {
        std::cout<<"Blurred image type is not CV_8UC1."<<std::endl;
        return FAILURE;
    }

    this->m_binaryImage = Mat::zeros(this->m_blurredImage.rows, this->m_blurredImage.cols, CV_8UC1);
    int threshold = getAdaptiveThreshold(this->m_blurredImage);

    for (int i = 0; i < this->m_image.rows; i++) {
        for (int j = 0; j < this->m_image.cols; j++) {
            this->m_binaryImage.at<uchar>(i,j) = this->m_blurredImage.at<uchar>(i,j) < threshold ? 0 : 255;
        }
    }

    return SUCCESS;

}



bool detectBlueRectangle(const Mat& inputImage, Rect& outRect) {
    if (inputImage.empty()) return false;

    Mat hsv;
    Mat possibleSection = inputImage(Rect(0, 0, inputImage.cols/3, inputImage.rows));

    cvtColor(possibleSection, hsv, COLOR_BGR2HSV);

    int minX = inputImage.cols, maxX = 0;
    int minY = inputImage.rows, maxY = 0;

    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            Vec3b hsvPixel = hsv.at<Vec3b>(y, x);

            if (ALPR::Util::isBlue(hsvPixel)) {
                minX = min(minX, x);
                maxX = max(maxX, x);
                minY = min(minY, y);
                maxY = max(maxY, y);
            }
        }
    }

    if (maxX > minX && maxY > minY) {
        outRect = Rect(Point(minX, minY), Point(maxX, maxY));
        return true;
    }

    return false;
}

/**
 *
 * @param image represents the candidate area applied on the image
 * @return SUCCESS if the area has the EU identifier. The EU identifier is the blue section at the beginning of each car-plate.
 * @return FAILURE if the image is empty or the candidate area doesn't have the EU blue section.
 */
int ALPR::ImageWorker::hasEUIdentifier(Mat image) {
    if (image.empty()) return FAILURE;

    Rect blueRect;
    if (detectBlueRectangle(image, blueRect)) {
        rectangle(image, blueRect, Scalar(0, 255, 0), 2);
        return SUCCESS;
    }

    return FAILURE;
}

bool isPlateAspectRatioValid(const Rect& boundingBox) {
    float aspectRatio = (float)boundingBox.width / boundingBox.height;
    return !(aspectRatio < 2.0f || aspectRatio > 5.0f);
}

int ALPR::ImageWorker::preProcess() {
    if (this->convertToGreyScale() == FAILURE) {
        std::cout << "Failed to convert to greyscale\n";
        return FAILURE;
    }

    if (this->applyBlur(Util::generateGaussianKernel(KERNEL_SIZE, 5)) == FAILURE) {
        std::cout << "Failed to apply blur\n";
        return FAILURE;
    }

    if (this->convertToBinary() == FAILURE) {
        std::cout << "Failed to convert to greyscale\n";
        return FAILURE;
    }

    return SUCCESS;
}

void ALPR::ImageWorker::process() {
    if (this->preProcess() == FAILURE) {
        std::cout << "Preprocessing failed\n";
        return;
    }

    Mat edges;
    Canny(this->m_blurredImage, edges, 75, 250);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    std::vector<Rect> candidates;
    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;

    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto contour : contours) {
        Rect bounding_rect = boundingRect(contour);
        Mat temp = this->m_image(bounding_rect);
        if (isPlateAspectRatioValid(bounding_rect) && hasEUIdentifier(temp) == SUCCESS) {
            candidates.push_back(bounding_rect);
        }
    }

    if (candidates.empty()) return;

    Rect bestPlate = candidates[0];
    double maxArea = bestPlate.width * bestPlate.height;

    for (size_t i = 1; i < candidates.size(); i++) {
        double area = candidates[i].width * candidates[i].height;
        if (area > maxArea) {
            maxArea = area;
            bestPlate = candidates[i];
        }    }

    this->m_ROI = this->m_image(bestPlate);
    imshow("Original Image",     this->m_image);
    imshow("Region of interest", this->m_ROI);
    waitKey(0);
}

void ALPR::ImageWorker::previewPreProcess() const {
    imshow("Original",      this->m_image);
    imshow("Greyscale",     this->m_greyscaleImage);
    imshow("Blurred",       this->m_blurredImage);
    imshow("BW",            this->m_binaryImage);
    waitKey(0);
}
