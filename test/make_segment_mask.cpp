#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "watershed.hpp"

using namespace cv;
using std::string;

struct UserData {
    Point p;
    Mat* src;
    Mat* mask;
};

#define FG Scalar(0,255,0,255)
#define BG Scalar(0,0,255,255)

void make_watershed_mask(const Mat& src, Mat& mask) {
	watershed_wrapper(src, mask);

	Mat dest = Mat(src.size(), CV_8UC1, Scalar(255));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			auto x = mask.at<Vec4b>(i, j);
			if (x[1] != 255) {
				dest.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("dest", dest);
	imwrite("images/9_mask.jpg", dest);
}

void on_mouse(int event, int x, int y, int flags, void* _data) {
    UserData* data = (UserData*)_data;
    if (event == EVENT_LBUTTONDOWN) {
        data->p = Point(x, y);
    } else if (event == EVENT_RBUTTONDOWN) { // edit
		make_watershed_mask(*data->src, *data->mask);
    } else if (event == EVENT_MOUSEMOVE) {
        Point point(x, y);
        if (flags & EVENT_FLAG_SHIFTKEY) {
            line(*(data->mask), data->p, point, FG, 1, 8, 0); 
            data->p = point;
        } else if ((flags & EVENT_FLAG_CTRLKEY)) {
            line(*(data->mask), data->p, point, BG, 1, 8, 0);
            data->p = point;
        }
        cv::imshow("mask", *(data->mask));
    }
}

int main(int argc, char const *argv[]) {
    string path = argv[1];
    Mat src = imread(path);
    cvtColor(src, src, COLOR_RGB2RGBA);

	Mat mask = Mat::zeros(src.size(), CV_8UC4);
    UserData data = { Point(0,0), &src, &mask };

    imshow("image", src);
    imshow("mask", mask);

    setMouseCallback("mask", on_mouse, &data);

    waitKey(-1);

    return 0;
}
