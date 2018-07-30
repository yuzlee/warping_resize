#pragma once

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "util.hpp"

using std::string;
using namespace cv;
using namespace Eigen;

template<class T>
inline void write_csv(const string &path, const SparseMatrix<T> &data) {
    std::ofstream f(path);
    if (!f) {
        std::cout << "can not open " << path << std::endl;
        return;
    }

    for (int i = 0; i < data.rows(); i++) {
        for (int j = 0; j < data.cols() - 1; j++) {
            const auto &p = data.coeff(i, j);
            f << std::to_string(p) << ", ";
        }
        f << std::to_string(data.coeff(i, data.cols() - 1)) << '\n';
    }
    f.close();
}

template<class T>
inline void write_csv(const string &path, const Mat &data) {
    std::ofstream f(path);
    if (!f) {
        std::cout << "can not open " << path << std::endl;
        return;
    }

    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols - 1; j++) {
            const auto &p = data.at<T>(i, j);
            f << std::to_string(p) << ", ";
        }
        f << std::to_string(data.at<T>(i, data.cols - 1)) << '\n';
    }
    f.close();
}

template <class T>
inline void read_csv() {

}