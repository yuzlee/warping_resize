#include <iostream>

#include "../src/interpolation.hpp"
#include "../src/global_warping.hpp"

using namespace std;
using namespace cv;

int main() {
    Line a = {Point{430, 75}, Point{468, 75}};

    Point p;
    if (a.line_intersection(Point{431,60}, Point{432,92}, p)) {
        cout << p << endl;
    } else {
        cout << "not found" << endl;
    }

    Matx81d V(398, 62, 431, 60, 399, 93, 432, 92);
    Point2d p0 = {430, 75};
    auto T = compute_translation(V, p0);
    cout << "T:" << endl << T << endl;

    return 0;
}