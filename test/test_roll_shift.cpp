#include <iostream>

#include "../src/seam_carving.hpp"
#include "../src/util.hpp"

using namespace cv;
using namespace std;

void test_roll() {
    int data[4][3] = {
            {1,2,3},
            {4,5,6},
            {7,8,9},
            {10,11,12}
    };
    Mat x(Size(3, 4), CV_32S, data);

    Mat row_roll, col_roll;
    roll_row_shift(x, row_roll, 1);
    roll_col_shift(x, col_roll, -1);

    cout << "origin: " << endl << x << endl;
    // cout << "row roll: " << endl << row_roll << endl;
    // cout << "col roll: " << endl << col_roll << endl;

    Mat mask = Mat::zeros(x.size(), CV_8U);
    SeamCarver carver(x, mask);
    mask.convertTo(mask, CV_32S);

    Mat energy, tracking;
    carver.compute_forward_energy(x, mask, energy, tracking);

    cout << "energy: " << endl << energy << endl;
}

int main(int argc, char* argv[]) {
    test_roll();

    return 0;
}