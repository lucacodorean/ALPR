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

        void addROI(const cv::Rect&);

        static int runValidation();
    private:
        cv::Mat m_image;
        cv::Mat m_greyscaleImage;
        cv::Mat m_blurredImage;
        cv::Mat m_binaryImage;
        cv::Mat m_ROI;

        inline static std::vector<cv::Rect> m_validation_ROIs = {
            cv::Rect(cv::Point(88, 139), cv::Point(156, 152)),
            cv::Rect(cv::Point(53, 118), cv::Point(172, 146)),
            cv::Rect(cv::Point(32, 91), cv::Point(183, 145)),
            cv::Rect(cv::Point(121, 75), cv::Point(185, 90)),
            cv::Rect(cv::Point(15, 68), cv::Point(258, 134)),
            cv::Rect(cv::Point(1098, 919), cv::Point(1541, 1073)),
            cv::Rect(cv::Point(177, 330), cv::Point(303, 368)),
            cv::Rect(cv::Point(102, 136), cv::Point(206, 163)),
            cv::Rect(cv::Point(467, 412), cv::Point(866, 508)),
            cv::Rect(cv::Point(99, 48), cv::Point(238, 97)),
        };
        static std::vector<cv::Rect> m_computedROIs;

        int convertToGreyScale();
        int convertToBinary();
        int applyBlur(vector<vector<double>>);

        static int hasEUIdentifier(cv::Mat image);
    };
}



#endif //IMAGEWORKER_H
