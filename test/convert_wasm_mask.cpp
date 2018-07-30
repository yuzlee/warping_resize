#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using std::string;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
	string path = argv[1];
	Mat src = imread(path, IMREAD_COLOR);

	Mat mask = Mat(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			auto m = src.at<Vec3b>(i, j);
			if (m[0] == 102 && m[1] == 255 && m[2] == 102) {
				mask.at<uchar>(i, j) = 255;
			} else {
				mask.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("wasm mask", src);
	imshow("mask", mask);

	waitKey(-1);

	return 0;
}
