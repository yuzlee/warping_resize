/**
 * NOTE: this file is a re-implementation of the `matlab` version,
 * ref: https://github.com/Jamie725/Rectangling-Panoramic-Images-Via-Warping
 */

#pragma once

#include <opencv2/core.hpp>

#include "util.hpp"
#include "mesh.hpp"

using namespace cv;

inline Matx28d compute_translation(const Matx81d &V_q, const Vec2d &p) {
    assert(V_q.rows == 8 && V_q.cols == 1);

    auto v1 = Point2d(V_q(0, 0), V_q(1, 0));
    auto v2 = Point2d(V_q(2, 0), V_q(3, 0));
    auto v3 = Point2d(V_q(4, 0), V_q(5, 0));
    auto v4 = Point2d(V_q(6, 0), V_q(7, 0));

    auto v21 = v2 - v1;
    auto v31 = v3 - v1;
    auto v41 = v4 - v1;
    auto p1 = Point2d(p[0], p[1]) - v1;

    auto a1 = v31.x;
    auto a2 = v21.x;
    auto a3 = v41.x - v21.x - v31.x;

    auto b1 = v31.y;
    auto b2 = v21.y;
    auto b3 = v41.y - v21.y - v31.y;

    auto px = p1.x;
    auto py = p1.y;

    double t1n, t2n;

    if (a3 == 0 && b3 == 0) {
        Matx21d t = Matx22d(v31.x, v21.x, v31.y, v21.y) * p1;
        t1n = t(0, 0);
        t2n = t(1, 0);
    } else {
        auto a = (b2 * a3 - a2 * b3);
        auto b = (-a2 * b1 + b2 * a1 + px * b3 - a3 * py);
        auto c = px * b1 - py * a1;
        if (a == 0) {
            t2n = -c / b;
        } else {
            t2n = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
        }
        if (abs(a1 + t2n * a3) <= 1e-6) {
            t1n = (py - t2n * b2) / (b1 + t2n * b3);
        } else {
            t1n = (px - t2n * a2) / (a1 + t2n * a3);
        }
    }

    auto m1 = v1 + t1n * (v3 - v1);
    auto m4 = v2 + t1n * (v4 - v2);

    auto v1w = 1 - t1n - t2n + t1n * t2n;
    auto v2w = t2n - t1n * t2n;
    auto v3w = t1n - t1n * t2n;
    auto v4w = t1n * t2n;

    return {v1w, 0, v2w, 0, v3w, 0, v4w, 0,
            0, v1w, 0, v2w, 0, v3w, 0, v4w};
}

inline void warp_mesh(const Mat &src, const Mesh &mesh, const Mesh &opt_mesh, Mat &target) {
    target.create(src.size(), CV_8UC3);
    Mat val = Mat::zeros(src.size(), CV_64FC3);
    Mat count = Mat::zeros(src.size(), CV_32S);

    Rect area{0, 0, src.cols, src.rows};

    for (int i = 0; i < mesh.quad_rows; i++) {
        for (int j = 0; j < mesh.quad_cols; j++) {
            const auto &V_origin = mesh.V(i, j);
            const auto &V_opt = opt_mesh.V(i, j);

            int min_x, max_x, min_y, max_y;
            opt_mesh.quad_xy({j, i}, min_x, max_x, min_y, max_y);
            int xn = 4 * (max_x - min_x);
            int yn = 4 * (max_y - min_y);
            double t1n = 0, t2n = 0;
            for (int y = 0; y <= yn; y++) {
                t1n = 1.0 / yn * y;
                for (int x = 0; x <= xn; x++) {
                    t2n = 1.0 / xn * x;

                    auto v1w = 1 - t1n - t2n + t1n * t2n;
                    auto v2w = t2n - t1n * t2n;
                    auto v3w = t1n - t1n * t2n;
                    auto v4w = t1n * t2n;
                    Matx28d T{v1w, 0, v2w, 0, v3w, 0, v4w, 0,
                              0, v1w, 0, v2w, 0, v3w, 0, v4w};

                    Matx21d _p_out = T * V_opt;
                    Matx21d _p_src = T * V_origin;
                    Point p_out{(int) lround(_p_out(0, 0)), (int) lround(_p_out(1, 0))};
                    Point p_src{(int) lround(_p_src(0, 0)), (int) lround(_p_src(1, 0))};

                    if (p_out.inside(area) && p_src.inside(area)) {
                        auto &v = val.at<Vec3d>(p_out);
                        auto &s = src.at<Vec3b>(p_src);
                        v[0] += s[0];
                        v[1] += s[1];
                        v[2] += s[2];
                        count.at<int>(p_out)++;
                    }
                }
            }
        }
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            auto &t = target.at<Vec3b>(i, j);
            auto &v = val.at<Vec3d>(i, j);
            auto c = (double) count.at<int>(i, j);

            t[0] = (uchar) lround(v[0] / c);
            t[1] = (uchar) lround(v[1] / c);
            t[2] = (uchar) lround(v[2] / c);
        }
    }
}


