#include <iostream>

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "seam_carving.hpp"
#include "global_warping.hpp"

#include "timer.hpp"

using namespace std;
using namespace cv;

inline void compute_color_threshold_mask(const Mat &src, Mat &mask, int threshold = 253) {
    mask.create(src.size(), CV_8U);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            auto &s = src.at<Vec3b>(i, j);
            auto &m = mask.at<uchar>(i, j);
            if (s[0] >= threshold && s[1] >= threshold && s[2] >= threshold) {
                m = 0xff;
            } else {
                m = 0x00;
            }
        }
    }
}

void process(int argc, char *argv[]) {
    Timer timer;

    const string keys =
            "{help h      |     | help information }"
            "{@image      |     | panoramic image }"
            "{@mask       |.    | the mask(0xff) for the panoramic image, color > (253,253,253) will be used if empty }"
            "{iteration i |10   | repeat i times of the optimization}"
            "{cols c      |10   | mesh cols}"
            "{rows r      |5    | mesh rows}"
            "{lambda_l    |100  | lambda for line energy}"
            // "{lambda_b |1e8  | lambda for boundary energy}"
            "{cache       |     | cache the local warping result}"
            "{scale       |1    | the scale for the input}";

    CommandLineParser parser(argc, argv, keys);
    parser.about("A re-implementation of the `Rectangling panoramic images via warping`");

    if (!parser.check() || parser.has("help")) {
        parser.printErrors();
        parser.printMessage();
        return;
    }

    OptimizationOption option{};
    option.nums_iteration = parser.get<int>("iteration");
    option.mesh_cols = parser.get<int>("cols");
    option.mesh_rows = parser.get<int>("rows");
    option.lambda_l = parser.get<double>("lambda_l");
    // option.lambda_b = parser.get<double>("cols");
    option.lambda_b = 1e8;

    bool use_cached_disp = parser.has("cache");

    string image_path = parser.get<string>(0);
    string mask_path = parser.get<string>(1);

    double scale = parser.get<double>("scale");

    Mat src = imread(image_path);
    if (src.empty()) {
        cout << "invalid image: " << image_path << endl;
        parser.printMessage();
        return;
    }

    Mat mask;
    if (mask_path != ".") {
        mask = imread(mask_path, IMREAD_GRAYSCALE);
    } else {
        compute_color_threshold_mask(src, mask, 253);
    }

    if (mask.empty()) {
        cout << "invalid image mask: " << mask_path << endl;
        return;
    }

    Mat disp;

    cout << "start" << endl;
    if (abs(scale - 1) > 1e-4) {
        resize(src, src, Size(), scale, scale);
        resize(mask, mask, Size(), scale, scale);
    }

    // ******** local warping ***********
    string disp_path = image_path + ".disp.yml";
    if (use_cached_disp) {
        FileStorage fs(disp_path, FileStorage::READ);
        fs["disp"] >> disp;
    } else {
        timer.reset();

        auto carver = SeamCarver(src, mask);
        carver.local_warping(disp);

        timer.tick("local warping ok");

        FileStorage fs(disp_path, FileStorage::WRITE);
        fs << "disp" << disp;
        fs.release();
    }

    // ******** global warping ***********
    timer.reset();

    auto optimizer = EnergyOptimizer(src, mask);
    optimizer.global_warping(disp, option);

    timer.tick("global warping ok");

    // ******** generate ***********
    timer.reset();

    Mat target;
    optimizer.generate(target);

    timer.tick("generate ok");

    // ******** show ***********
    imshow("target", target);

    waitKey(-1);
}

int main(int argc, char *argv[]) {
    process(argc, argv);

    return 0;
}