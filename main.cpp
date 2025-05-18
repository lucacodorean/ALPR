#include <opencv2/highgui.hpp>
#include "src/ImageWorker.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace ALPR;

void test() {
    for (int i = 1; i<=12; i++) {
        char path[256];
        sprintf_s(path, "E:\\UNIVERSITY\\AN III\\Procesare_de_imagini\\Laboratoare\\Proiect\\images\\test_set\\test_%d.bmp", i);
        ImageWorker worker = ImageWorker(path);
        worker.process();
        cv::waitKey(0);
    }
}

void val() {
    ofstream fout("data.csv");
    fout<<"index,iou\n";

    static std::vector m_validation_ROIs = {
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

    for (int i = 0; i<10; i++) {
        char path[256];
        sprintf_s(path, "E:\\UNIVERSITY\\AN III\\Procesare_de_imagini\\Laboratoare\\Proiect\\images\\val_set\\val_%d.jpeg", i+1);
        ImageWorker worker = ImageWorker(path, m_validation_ROIs[i]);
        worker.process();
        fout<<i<<","<<worker.validate()<<"\n";
        cv::waitKey(0);
    }

    fout.close();
}

int main() {

    int command;
    std::cout<<"Insert the command you want to do: 1 for test or 2 for val. \nCOMMAND: ";
    std::cin>>command;

    switch (command) {
        case 1:
            test();
            break;
        case 2:
            val();
            break;
        default:
            return 0;
    }
    return 0;
}