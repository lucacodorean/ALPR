#include <opencv2/highgui.hpp>

#include "src/ImageWorker.h"
#include <iostream>

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
    for (int i = 1; i<=10; i++) {
        char path[256];
        sprintf_s(path, "E:\\UNIVERSITY\\AN III\\Procesare_de_imagini\\Laboratoare\\Proiect\\images\\val_set\\val_%d.jpeg", i);
        ImageWorker worker = ImageWorker(path);
        worker.process();
        cv::waitKey(0);
    }

    ImageWorker::runValidation();
}

int main() {

    string command;
    std::cout<<"Insert the command you want to do: test or val. \nCOMMAND: ";
    std::cin>>command;

    if (command == "test")      test();
    else if (command == "val")  val();
    return 0;
}