#include "src/ImageWorker.h"

using namespace std;
using namespace ALPR;

int main() {
    for (int i = 1; i<=10; i++) {
        char path[256];
        sprintf(path, "E:\\UNIVERSITY\\AN III\\Procesare_de_imagini\\Laboratoare\\Proiect\\images\\test_set\\test_%d.png", i);
        ImageWorker worker = ImageWorker(path);
        worker.process();
    }

    return 0;
}