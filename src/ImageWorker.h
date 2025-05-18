//
// Created by codor on 4/6/2025.
//

#ifndef IMAGEWORKER_H
#define IMAGEWORKER_H

#include <string>
#include <opencv2/core/mat.hpp>
using namespace std;

namespace ALPR {
    class ImageWorker {
    public:
        explicit ImageWorker(const string&);
        explicit ImageWorker(const string&, cv::Rect);
        int preProcess();
        void previewPreProcess() const;
        void process();
        double validate() const;

        int getWidth() const;
        int getHeight() const;

    private:
        cv::Mat m_image;
        cv::Mat m_greyscaleImage;
        cv::Mat m_blurredImage;
        cv::Mat m_binaryImage;
        cv::Mat m_ROI;
        cv::Rect m_computedROI;
        cv::Rect m_validationROI;

        int convertToGreyScale();
        int convertToBinary();
        int applyBlur(vector<vector<double>>);

        static int hasEUIdentifier(cv::Mat image);
    };
}



#endif //IMAGEWORKER_H
