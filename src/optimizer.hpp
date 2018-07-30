#pragma once

#include "util.hpp"
#include "line.hpp"
#include "mesh.hpp"

using namespace cv;

const double MIN_VALUE = 1e-8;

class ShapeOptimizer {
public:
    typedef Table2D<Matx88d> ShapeMatrix;

    explicit ShapeOptimizer(const Mesh &mesh) : mesh(mesh) {
        A_term.init(mesh.size());
    }

    /**
     * E_s = A_term * V_q
     * @return MxNx8x8 tensor
     */
    const ShapeMatrix &shape_energy_term() {
        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                const Point &q0 = mesh.at(i, j);
                const Point &q1 = mesh.at(i, j + 1);
                const Point &q2 = mesh.at(i + 1, j);
                const Point &q3 = mesh.at(i + 1, j + 1);

                double A_data[] = {
                        (double) q0.x, (double) -q0.y, 1, 0,
                        (double) q0.y, (double) q0.x, 0, 1,
                        (double) q1.x, (double) -q1.y, 1, 0,
                        (double) q1.y, (double) q1.x, 0, 1,
                        (double) q2.x, (double) -q2.y, 1, 0,
                        (double) q2.y, (double) q2.x, 0, 1,
                        (double) q3.x, (double) -q3.y, 1, 0,
                        (double) q3.y, (double) q3.x, 0, 1
                };
                Matx84d A_q(A_data);
                A_term.at(i, j) = A_q * (A_q.t() * A_q).inv() * A_q.t() - Matx88d::eye();
            }
        }

        return A_term;
    }

    void make_block_diag(SMatrixd &K, double lambda, int r0, int c0) {
        shape_energy_term();

        int n = mesh.quad_rows * mesh.quad_cols;

        lambda /= (double) n;

        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                auto k = i * mesh.quad_cols + j;
                const Matx88d &A = A_term.at(i, j);
                for (int r = 0; r < A.rows; r++) {
                    for (int c = 0; c < A.cols; c++) {
                        const auto e = A(r, c) * lambda;
                        if (abs(e) < MIN_VALUE) { continue; }
                        K.insert(r0 + 8 * k + r, c0 + 8 * k + c) = e;
                    }
                }
            }
        }
    }

private:
    const Mesh &mesh;

    Table2D<Matx88d> A_term;
};

template<int M = 50>
class LineOptimizer {
public:
    typedef vector<Line> LineList;
    typedef Table2D<Mat> LineMatrix;

    explicit LineOptimizer(const Mesh &mesh) : mesh(mesh) {
        distortion.init(mesh.size());
        mesh_lines.init(mesh.size());
    }

    /**
     * detect line segment in original image, and warp the lines using displacement filed
     * @param disp displacement filed
     * @param lines[out] line vector
     */
    void find_and_init_lines(const Mat &src) {
        LsdLineList list;
        detect_line_segment(src, list);

#ifdef _DEBUG
        // mesh.show_mesh_quad(src, res);
        // for (int i = 0; i < list.length(); i++) {
        //     Line line{list[i].start(), list[i].end()};
        //     cv::line(res, line.start, line.end, Scalar(0, 0, 255), 2);
        // }
        // imshow("original lines", res);
        // waitKey(-1);
        // mesh.show_mesh_quad(src, res);
#endif

        // cut all the detected line segments with the edges of the input mesh
        // and compute initial translation matrix for each line
        for (int i = 0; i < list.length(); i++) {
            Line line{list[i].start(), list[i].end()};
            Point start_id, end_id;
            mesh.which_quad(line.start, start_id);
            mesh.which_quad(line.end, end_id);

            const Point invalid_p{-1, -1};
            if (start_id == invalid_p || end_id == invalid_p) {
#ifdef _DEBUG
                // Mat _res;
                // mesh.show_mesh_quad(src, _res);
                // circle(_res, line.start, 4, Scalar(0, 0, 255), -1);
                // circle(_res, line.end, 4, Scalar(255, 0, 0), -1);
                // imshow("bad point", _res);
                // waitKey(-1);
#endif
                continue;
            }

#ifdef _DEBUG
            // Mat _res = src.clone();
            // // mesh.show_mesh_quad(src, _res);
            // mesh.show_mesh_quad(src, start_id, _res);
            // mesh.show_mesh_quad(src, end_id, _res);
            // circle(_res, line.start, 4, Scalar(0, 0, 255), -1);
            // circle(_res, line.end, 4, Scalar(255, 0, 0), -1);
            // imshow("good point", _res);
            // waitKey(-1);
#endif

            auto current_id = start_id;
            auto endpoint_start = line.start;
            auto endpoint_end = line.end;

            const Point id_offset[4] = {
                    Point(0, -1), // q1<->q2, top
                    Point(1, 0), // q2<->q4, right
                    Point(0, 1), // q4<->q3, bottom
                    Point(-1, 0) // q3<->q1, left
            };
            auto offset_idx = 0;

            while (mesh.contains_quad(current_id)) {
                if (current_id == end_id) { // if the line is ended at the current quad
                    endpoint_end = line.end;

                    // add line to mesh_lines
                    add_line(current_id, endpoint_start, endpoint_end);
                    break;
                } else { // the line must intersect with (only) one edge of the current quad
                    auto &quad = mesh.quad(current_id);
                    auto &q1 = quad.at<Point>(0, 0); // left-top
                    auto &q2 = quad.at<Point>(0, 1); // right-top
                    auto &q3 = quad.at<Point>(1, 0); // left-bottom
                    auto &q4 = quad.at<Point>(1, 1); // right-bottom

                    const Point *q_list[5] = {&q1, &q2, &q4, &q3, &q1}; // slide point list

                    bool is_intersected = false;
                    for (int k = 0; k < 4; k++) {
                        if (Line{endpoint_start, line.end}.line_intersection(*q_list[k], *q_list[k + 1],
                                                                             endpoint_end)) {
                            auto diff = endpoint_start - endpoint_end;
                            if (diff.dot(diff) < 4) { continue; }

                            is_intersected = true;
                            offset_idx = k;

                            // add line to mesh_lines
                            // cv::line(res, *q_list[k], *q_list[k + 1], Scalar(255, 0, 0));
                            add_line(current_id, endpoint_start, endpoint_end);

                            break;
                        }
                    }
                    if (!is_intersected) { break; }

                    current_id += id_offset[offset_idx];
                    endpoint_start = endpoint_end;

                    // if the rest of line is too short
                    auto diff = endpoint_start - line.end;
                    if (diff.dot(diff) < 4) {
                        break;
                    }
                }
            }
        }

#ifdef _DEBUG
        // show_lines(src);
#endif
    }

    /**
     * E_l = C_q * e = C_q * T * V, let C_term = C_q * T
     * @return MxNx(Lx2)x8 tensor
     */
    const LineMatrix &line_distortion_term() {
        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                auto &lines = mesh_lines.at(i, j);
                if (lines.empty()) { continue; }
                auto &C_term = distortion.at(i, j);
                C_term.create(2 * (int) lines.size(), 8, CV_64F); // e: 2x8 matrix
                int line_idx = 0;
                for (Line &line: lines) {
                    auto theta = line.theta;
                    // rotation matrix
                    Matx22d R(cos(theta), -sin(theta), sin(theta), cos(theta));
                    Vec2d e_hat = line.difference();
                    Matx22d C = R * e_hat * (e_hat.t() * e_hat).inv() * e_hat.t() * R.t() - Matx22d::eye();
                    Matx28d e = line.translation_difference();

                    Mat Ce = Mat(C * e);
                    Ce.copyTo(C_term(Range(2 * line_idx, 2 * line_idx + 2), Range::all()));

                    // std::cout << "Ce:" << std::endl << Ce << std::endl;
                    // std::cin.ignore();

                    line_idx++;
                }
            }
        }

        return distortion;
    }

    void make_block_diag(SMatrixd &K, double lambda, int r0, int c0) {
        line_distortion_term();

        lambda /= (double) line_count;

        int row_offset = 0;
        int col_offset = 0;
        const int col_offset_base = 8;

        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                const auto &C = distortion.at(i, j);
                if (C.empty()) {
                    col_offset += col_offset_base;
                } else {
                    assert(C.cols == col_offset_base);

                    for (int r = 0; r < C.rows; r++) {
                        for (int c = 0; c < C.cols; c++) {
                            const auto e = C.at<double>(r, c) * lambda;
                            if (abs(e) < MIN_VALUE) { continue; }
                            K.insert(r0 + row_offset + r, c0 + col_offset + c) = e;
                        }
                    }

                    row_offset += C.rows;
                    col_offset += col_offset_base;
                }
            }
        }
    }

    /**
     * fix V, update theta
     * @param opt_mesh
     */
    void update_line_theta(const Mesh &opt_mesh) {
        double theta_bin[M] = {0};
        int theta_bin_count[M] = {0};

        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                for (Line &line: mesh_lines.at(i, j)) {
                    int id = line.group_id(M);
                    theta_bin[id] += line.get_rotation_angle(opt_mesh.V(i, j));
                    theta_bin_count[id]++;
                }
            }
        }

        // compute mean theta for each bin
        for (int i = 0; i < M; i++) {
            theta_bin[i] /= theta_bin_count[i];
        }

        // update delta for each line
        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                auto &lines = mesh_lines.at(i, j);
                for (Line &line:lines) {
                    int id = line.group_id(M);
                    line.theta = theta_bin[id];
                }
            }
        }
    }

    inline int get_nums_line() const {
        return line_count;
    }

#ifdef _DEBUG
    void show_lines(const Mat &src) {
            for (int i = 0; i < mesh.quad_rows; i++) {
                for (int j = 0; j < mesh.quad_cols; j++) {
                    auto &lines = mesh_lines.at(i, j);
                    for (Line &line: lines) {
                        cv::line(res, line.start, line.end, Scalar(0, 0, 255), 2);
                    }
                }
            }

            imshow("lines in image", res);
            waitKey(-1);
        }
#endif

private:
    const Mesh &mesh;

    Table2D<Mat> distortion;
    Table2D<LineList> mesh_lines;

#ifdef _DEBUG
    Mat res; // show mesh and lines
#endif

    int line_count = 0;

    inline void add_line(const Point &quad_id, const Point &s, const Point &e) {
        auto segment = Line::init_line(s, e, mesh.V(quad_id));
        mesh_lines.at(quad_id).push_back(segment);

        line_count++;

#ifdef _DEBUG
        // cv::line(res, s, e, Scalar(0, 0, 255), 2);
        // imshow("line in mesh", res);
        // waitKey(-1);
#endif
    }
};
