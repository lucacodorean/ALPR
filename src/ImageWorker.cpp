//
// Created by codor on 4/6/2025.
//

#include "ImageWorker.h"

#include "util.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


int ALPR::ImageWorker::getHeight() const {
    return this->m_image.rows;
}

int ALPR::ImageWorker::getWidth() const {
    return this->m_image.cols;
}


ALPR::ImageWorker::ImageWorker(const string& path) {
    if (path.empty()) {
        std::cout<<"Can't operate with this path.";
        return;
    }

    this->m_image = cv::imread(path, IMREAD_COLOR);
}


ALPR::ImageWorker::ImageWorker(const string& path, Rect validationROI) {
    if (path.empty()) {
        std::cout<<"Can't operate with this path.";
        return;
    }

    this->m_image = cv::imread(path, IMREAD_COLOR);
    this->m_validationROI = validationROI;
}

int ALPR::ImageWorker::convertToGreyScale() {

    if (this->m_image.empty()) {
        std::cout<<"Image not set-up yet."<<std::endl;
        return FAILURE;
    }

    this->m_greyscaleImage = Mat::zeros(this->m_image.rows, this->m_image.cols, CV_8UC1);

    for (int i = 0; i<this->m_image.rows; i++) {
        for (int j = 0; j<this->m_image.cols; j++) {
            this->m_greyscaleImage.at<uchar>(i, j) =
                (this->m_image.at<Vec3b>(i, j)[BLUE_CHANNEL] + this->m_image.at<Vec3b>(i, j)[GREEN_CHANNEL] + this->m_image.at<Vec3b>(i, j)[RED_CHANNEL])/3 ;
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


int ALPR::ImageWorker::convertToBinary() {
    if (this->m_blurredImage.type() != CV_8UC1) {
        std::cout<<"Blurred image type is not CV_8UC1."<<std::endl;
        return FAILURE;
    }

    this->m_binaryImage = Mat::zeros(this->m_blurredImage.rows, this->m_blurredImage.cols, CV_8UC1);

    for (int i = 0; i < this->m_image.rows; i++) {
        for (int j = 0; j < this->m_image.cols; j++) {
            this->m_binaryImage.at<uchar>(i,j) = this->m_blurredImage.at<uchar>(i,j) < BW_THRESHOLD ? 0 : 255;
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

    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            Vec3b hsvPixel = hsv.at<Vec3b>(i,j);

            if (ALPR::Util::isBlue(hsvPixel)) {
                minX = min(minX, j);
                maxX = max(maxX, j);
                minY = min(minY, i);
                maxY = max(maxY, i);
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
        std::cout << "Failed to convert to binary\n";
        return FAILURE;
    }

    return SUCCESS;
}

void ALPR::ImageWorker::process() {
    if (this->preProcess() == FAILURE) {
        std::cout << "Preprocessing failed\n";
        return;
    }


    /*
     * Algoritmul Canny a fost implementat si in clasa Util, inca trebuie determinati parametri care sa se muleze pe acesta.
     * Algoritmul din biblioteca openCV permite setarea marimii kernelului sobel si o optimizare pentru gradient
     */

    Mat edges;
    Canny(this->m_blurredImage, edges, 75, 250, 3, true);
    edges = Util::closing(edges, Util::getNeighborhood(),1);

    std::vector<Rect> candidates;
    std::vector<Rect> allContours;
    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> contours;

    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto contour : contours) {
        Rect bounding_rect = boundingRect(contour);
        Mat temp = this->m_image(bounding_rect);
        if (isPlateAspectRatioValid(bounding_rect) && hasEUIdentifier(temp) == SUCCESS) {
            candidates.push_back(bounding_rect);
        }

        allContours.push_back(bounding_rect);
    }

    if (candidates.empty()) candidates = allContours;

    Rect bestPlate = candidates[0];
    double maxArea = bestPlate.width * bestPlate.height;

    for (size_t i = 1; i < candidates.size(); i++) {
        double area = candidates[i].width * candidates[i].height;
        if (area > maxArea) {
            maxArea = area;
            bestPlate = candidates[i];
        }
    }

    this->m_ROI = this->m_image(bestPlate);
    rectangle(this->m_image, this->m_validationROI, Scalar(0, 255, 255), 2);
    this->m_computedROI = bestPlate;
    imshow("Original Image", this->m_image);
    imshow("Region of interest", this->m_ROI);
}

void ALPR::ImageWorker::previewPreProcess() const {
    imshow("Original",      this->m_image);
    imshow("Greyscale",     this->m_greyscaleImage);
    imshow("Blurred",       this->m_blurredImage);
    imshow("BW",            this->m_binaryImage);
    waitKey(0);
}

double computeIOU(const Rect& box1, const Rect& box2) {
    cout<<box1<<" "<<box2<<endl;

    int intersectionArea = (box1 & box2).area();
    int unionArea = box1.area() + box2.area() - intersectionArea;
    return 1.0 * intersectionArea / unionArea;
}

double ALPR::ImageWorker::validate() const {
    double iou = computeIOU(this->m_computedROI, this->m_validationROI);
    std::cout << (iou>IOU_THRESHOLD ? "Positive" :  "False positive") <<" plate identified with IoU value: " << iou << "\n";
    return iou;
}
