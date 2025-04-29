#include <opencv2/highgui.hpp>

#include "src/ImageWorker.h"

using namespace std;
using namespace ALPR;

int main() {

    for (int i = 1; i<=12; i++) {
        char path[256];
        sprintf(path, "E:\\UNIVERSITY\\AN III\\Procesare_de_imagini\\Laboratoare\\Proiect\\images\\test_set\\test_%d.bmp", i);
        ImageWorker worker = ImageWorker(path);
        worker.process();
        cv::waitKey(0);
    }

    return 0;
}