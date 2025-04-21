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
        int preProcess();
        void previewPreProcess() const;
        void process();


    private:
        cv::Mat m_image;
        cv::Mat m_greyscaleImage;
        cv::Mat m_blurredImage;
        cv::Mat m_binaryImage;
        cv::Mat m_ROI;

        int convertToGreyScale();
        int convertToBinary();
        int applyBlur(vector<vector<double>>);
        static int hasEUIdentifier(cv::Mat image);
    };
}



#endif //IMAGEWORKER_H
