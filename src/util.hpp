#pragma once

#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Sparse>

using std::vector;

using namespace cv;
using namespace Eigen;

typedef Matx<double, 1, 8> Matx18d;
typedef Matx<double, 8, 1> Matx81d;
typedef Matx<double, 2, 8> Matx28d;
typedef Matx<double, 8, 4> Matx84d;
typedef Matx<double, 8, 8> Matx88d;
typedef SparseMatrix<double> SMatrixd;
typedef SparseVector<double> SVectord;

typedef SparseQR<SMatrixd, COLAMDOrdering<int>> QRSolver;

inline SMatrixd eigen_inv(const SMatrixd &A) {
    QRSolver solver;
    solver.compute(A);
    SMatrixd eye(A.rows(), A.cols());
    eye.setIdentity();
    return solver.solve(eye);
}

inline SMatrixd eigen_row_concat(const SMatrixd &A, const SMatrixd &B) {
    assert(A.cols() == B.cols());

    const auto A_rows = A.rows();
    const auto B_rows = B.rows();
    SMatrixd C(A_rows + B_rows, A.cols());

    for (int k = 0; k < A.outerSize(); ++k) {
        for (SMatrixd::InnerIterator it(A, k); it; ++it) {
            C.insert(it.row(), it.col()) = it.value();
        }
    }
    for (int k = 0; k < B.outerSize(); ++k) {
        for (SMatrixd::InnerIterator it(B, k); it; ++it) {
            C.insert(A_rows + it.row(), it.col()) = it.value();
        }
    }

    C.makeCompressed();

    return C;
}

template<class T>
class Table2D {
public:
    Table2D() = default;

    void init(const Size &size) {
        if (data != nullptr) {
            delete[] data;
        }
        this->size = size;
        data = new T[size.width * size.height];
    }

    ~Table2D() {
        if (data != nullptr) {
            delete[] data;
        }
    }

    inline T &at(int i, int j) {
        assert(i >= 0 && i < size.height && j >= 0 && j < size.width);
        return data[i * size.width + j];
    }

    inline const T &at(int i, int j) const {
        assert(i >= 0 && i < size.height && j >= 0 && j < size.width);
        return data[i * size.width + j];
    }

    inline T &at(const Point &p) {
        return at(p.y, p.x);
    }

    inline const T &at(const Point &p) const {
        return at(p.y, p.x);
    }

private:
    Size size;
    T *data = nullptr;
};


namespace cv {
    inline void roll_row_shift(const Mat &src, Mat &dest, int shift_m_rows) {
        dest.create(src.size(), src.type());

        int m = shift_m_rows;
        int rows = src.rows;
        if (m % rows == 0) {
            src.copyTo(dest);
            return;
        }

        if (m > 0) {
            src(Range(rows - m, rows), Range::all()).copyTo(dest(Range(0, m), Range::all()));
            src(Range(0, rows - m), Range::all()).copyTo(dest(Range(m, rows), Range::all()));
        } else {
            src(Range(0, -m), Range::all()).copyTo(dest(Range(rows + m, rows), Range::all()));
            src(Range(-m, rows), Range::all()).copyTo(dest(Range(0, rows + m), Range::all()));
        }
    }

    inline void roll_col_shift(const Mat &src, Mat &dest, int shift_n_cols) {
        dest.create(src.size(), src.type());

        int n = shift_n_cols;
        int cols = src.cols;
        if (n % cols == 0) {
            src.copyTo(dest);
            return;
        }

        if (n > 0) {
            src(Range::all(), Range(cols - n, cols)).copyTo(dest(Range::all(), Range(0, n)));
            src(Range::all(), Range(0, cols - n)).copyTo(dest(Range::all(), Range(n, cols)));
        } else {
            src(Range::all(), Range(0, -n)).copyTo(dest(Range::all(), Range(cols + n, cols)));
            src(Range::all(), Range(-n, cols)).copyTo(dest(Range::all(), Range(0, cols + n)));
        }
    }

    /**
     * calculate the argmin vector for matrices
     * @tparam _Ty
     * @param a row-vector
     * @param b row-vector
     * @param c row-vector
     * @param dest the dest row-vector
     */
    template<class _Ty>
    inline void triple_min(const Mat &a, const Mat &b, const Mat &c, Mat &min_idx, Mat &min_val) {
        assert(a.cols == b.cols && b.cols == c.cols);

        min_idx.create(1, a.cols, CV_32S);
        min_val.create(1, a.cols, a.type());
        for (int j = 0; j < a.cols; j++) {
            _Ty vals[3] = {a.at<_Ty>(0, j), b.at<_Ty>(0, j), c.at<_Ty>(0, j)};
            _Ty _min_val = vals[1];
            int _min_idx = 1;
            for (int k = 0; k < 3; k++) {
                if (vals[k] < _min_val) {
                    _min_val = vals[k];
                    _min_idx = k;
                }
            }
            min_idx.at<int>(0, j) = _min_idx;
            min_val.at<_Ty>(0, j) = _min_val;
        }
    }

    void show_energy_map(const Mat &energy_map) {
        double _min, _max;
        minMaxIdx(energy_map, &_min, &_max);

        double log_a = log(_max) / 255.0;
        double delta = _min - 1;

        Mat dest(energy_map.size(), CV_8U);
        for (int i = 0; i < energy_map.rows; i++) {
            for (int j = 0; j < energy_map.cols; j++) {
                auto x = energy_map.at<int>(i, j);
                auto e = (uchar) (log(x - delta) / log_a);
                dest.at<uchar>(i, j) = e;
            }
        }

        imshow("energy map", dest);
    }

    inline bool is_mask(uchar v) {
        return v > 128;
    }
}
