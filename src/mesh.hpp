#pragma once

#include "util.hpp"

using namespace cv;

class Mesh {
public:
    Mesh() = default;

    void init(const Size &size) {
        data.create(size, CV_32SC2);

        vertex_rows = data.rows; // horizontal `quad` amount
        vertex_cols = data.cols;
        quad_rows = vertex_rows - 1;
        quad_cols = vertex_cols - 1;
    }

    /**
     *
     * @param src_size
     * @param size size.width is the number of horizontal vertexes, ...
     */
    Mesh &init_from_image(const Size &src_size, const Size &size) {
        data.create(size, CV_32SC2);
        this->vertex_rows = data.rows; // horizontal `quad` amount
        this->vertex_cols = data.cols;
        this->quad_rows = this->vertex_rows - 1;
        this->quad_cols = this->vertex_cols - 1;

        this->dx = src_size.width / (double) (size.width - 1);
        this->dy = src_size.height / (double) (size.height - 1);
        for (int i = 0; i < size.height; i++) {
            auto y = (int) lround(dy * i);
            if (y > src_size.height - 1) { // align the grid to bottom
                y = src_size.height - 1;
            }
            for (int j = 0; j < size.width; j++) {
                auto x = (int) lround(dx * j);
                if (x > src_size.width - 1) { // align the grid to right
                    x = src_size.width - 1;
                }

                data.at<Vec2i>(i, j) = Vec2i(x, y); // Point(x, y)
            }
        }

        return *this;
    }

    /**
     *
     * @param disp displacement matrix
     */
    void displace(const Mat &disp) {
        for (int i = 0; i < data.rows; i++) {
            for (int j = 0; j < data.cols; j++) {
                const auto &p = data.at<Point>(i, j);
                data.at<Vec2i>(i, j) = disp.at<Vec2i>(p);
            }
        }
    }

    inline void offset() {
        for (int i = 0; i < data.rows; i++) {
            for (int j = 0; j < data.cols; j++) {
                auto &p = data.at<Point>(i, j);
                p.x++;
                p.y++;
            }
        }
    }

    inline void offset_back() {
        for (int i = 0; i < data.rows; i++) {
            for (int j = 0; j < data.cols; j++) {
                auto &p = data.at<Point>(i, j);
                p.x--;
                p.y--;
            }
        }
    }

    inline const Mat quad(int i, int j) const {
        return data(Range(i, i + 2), Range(j, j + 2));
    }

    inline const Mat quad(const Point &p) const {
        return quad(p.y, p.x);
    }

    inline const Matx81d V(int i, int j) const {
        const auto &q1 = data.at<Point>(i, j);
        const auto &q2 = data.at<Point>(i, j + 1);
        const auto &q3 = data.at<Point>(i + 1, j);
        const auto &q4 = data.at<Point>(i + 1, j + 1);
        const Point *q_list[] = {&q1, &q2, &q3, &q4};
        double v_data[8];
        for (int k = 0; k < 4; k++) {
            v_data[2 * k] = (double) q_list[k]->x;
            v_data[2 * k + 1] = (double) q_list[k]->y;
        }
        return Matx81d(v_data);
    }

    inline const Matx81d V(const Point &p) const {
        return V(p.y, p.x);
    }

    /**
     * compute the target quad of a point in the rect mesh
     * @param p
     * @param id
     */
    inline void which_quad(const Point &p, Point &id) const {
        // const auto &t = inv_disp.at<Point>(p);
        // id.y = (int) (t.y / dy);
        // id.x = (int) (t.x / dx);

        for (int i = 0; i < quad_rows; i++) {
            for (int j = 0; j < quad_cols; j++) {
                if (inside(Point(j, i), p)) {
                    id.x = j;
                    id.y = i;
                    return;
                }
            }
        }
        id.x = -1;
        id.y = -1;
    }

    /**
     * (8*qx*qy)x(2*vx*vy)
     * @return
     */
    inline SMatrixd Q() const {
        SMatrixd Q(8 * quad_rows * quad_cols, 2 * vertex_rows * vertex_cols);
        for (int i = 0; i < quad_rows; i++) {
            for (int j = 0; j < quad_cols; j++) {
                auto quad_id = (i * quad_cols + j) * 8;
                auto lt_id = (i * vertex_cols + j) * 2;

                Q.insert(quad_id, lt_id) = 1;
                Q.insert(quad_id + 1, lt_id + 1) = 1;

                Q.insert(quad_id + 2, lt_id + 2) = 1;
                Q.insert(quad_id + 3, lt_id + 3) = 1;

                Q.insert(quad_id + 4, lt_id + vertex_cols * 2) = 1;
                Q.insert(quad_id + 5, lt_id + vertex_cols * 2 + 1) = 1;

                Q.insert(quad_id + 6, lt_id + vertex_cols * 2 + 2) = 1;
                Q.insert(quad_id + 7, lt_id + vertex_cols * 2 + 3) = 1;
            }
        }

        Q.makeCompressed();

        return Q;
    }

    inline const Point &at(int i, int j) const {
        return data.at<Point>(i, j);
    }

    inline Point &at(int i, int j) {
        return data.at<Point>(i, j);
    }

    inline const Size size() const {
        return Size(quad_cols, quad_rows);
    }

    inline const Mat &mat() const {
        return data;
    }

    inline bool contains_quad(const Point &p) const {
        return 0 <= p.x && p.x < quad_cols && 0 <= p.y && p.y < quad_rows;
    }

    int vertex_cols = 0, vertex_rows = 0;
    int quad_cols = 0, quad_rows = 0;

    void show_mesh(const Mat &src, const std::string &name) {
        Mat res = src.clone();

        for (int i = 0; i < vertex_rows; i++) {
            for (int j = 0; j < vertex_cols; j++) {
                Point p = data.at<Point>(i, j);
                circle(res, p, 5, Scalar(0, 0, 255), -1);
            }
        }

        imshow(name, res);
        waitKey(-1);
    }

    void show_mesh_quad(const Mat &src, Mat &res) const {
        res = src.clone();

        for (int i = 0; i < quad_rows; i++) {
            for (int j = 0; j < quad_cols; j++) {
                auto p1 = data.at<Point>(i, j);
                auto p2 = data.at<Point>(i, j + 1);
                auto p3 = data.at<Point>(i + 1, j);
                cv::line(res, p1, p2, Scalar(0, 255, 0));
                cv::line(res, p1, p3, Scalar(0, 255, 0));
            }
        }

        for (int i = 0; i < quad_rows; i++) {
            cv::line(res, data.at<Point>(i, quad_cols), data.at<Point>(i + 1, quad_cols), Scalar(0, 255, 0));
        }
        for (int j = 0; j < quad_cols; j++) {
            cv::line(res, data.at<Point>(quad_rows, j), data.at<Point>(quad_rows, j + 1), Scalar(0, 255, 0));
        }
    }

    void show_mesh_quad(const Mat &src, const Point &quad_id, Mat &res) const {
        // res = src.clone();

        const auto &p = quad_id;
        const auto &p1 = data.at<Point>(quad_id);
        const auto &p2 = data.at<Point>(Point{p.x + 1, p.y});
        const auto &p3 = data.at<Point>(Point{p.x, p.y + 1});
        const auto &p4 = data.at<Point>(Point{p.x + 1, p.y + 1});
        cv::line(res, p1, p2, Scalar(0, 255, 0));
        cv::line(res, p2, p4, Scalar(0, 255, 0));
        cv::line(res, p3, p4, Scalar(0, 255, 0));
        cv::line(res, p1, p3, Scalar(0, 255, 0));
    }

    inline void quad_xy(const Point &quad_id, int &min_x, int &max_x, int &min_y, int &max_y) const {
        const int q_i = quad_id.y, q_j = quad_id.x;
        const auto &q1 = data.at<Point>(q_i, q_j);
        const auto &q2 = data.at<Point>(q_i, q_j + 1);
        const auto &q3 = data.at<Point>(q_i + 1, q_j);
        const auto &q4 = data.at<Point>(q_i + 1, q_j + 1);

        min_x = std::min(q1.x, q3.x);
        min_y = std::min(q1.y, q2.y);
        max_x = std::min(q2.x, q4.x);
        max_y = std::min(q3.y, q4.y);
    }

private:
    Mat data;

    double dx = 0, dy = 0;

    inline bool inside(const Point &quad_id, const Point &p) const {
        const int nums_vertex = 4;

        const int q_i = quad_id.y, q_j = quad_id.x;
        const auto &q1 = data.at<Point>(q_i, q_j);
        const auto &q2 = data.at<Point>(q_i, q_j + 1);
        const auto &q3 = data.at<Point>(q_i + 1, q_j);
        const auto &q4 = data.at<Point>(q_i + 1, q_j + 1);
        const Point *vertexes[] = {&q1, &q2, &q4, &q3, &q1};

        auto min_x = std::min(q1.x, q3.x);
        auto min_y = std::min(q1.y, q2.y);
        auto max_x = std::min(q2.x, q4.x);
        auto max_y = std::min(q3.y, q4.y);

        if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) {
            return false;
        }

        double delta_sum = 0;
        for (int i = 0; i < nums_vertex; i++) {
            auto v1 = p - *vertexes[i];
            auto v2 = p - *vertexes[i + 1];
            double delta = atan2(v1.y, v1.x) - atan2(v2.y, v2.x);
            while (delta > CV_PI) {
                delta -= (2 * CV_PI);
            }
            while (delta < -CV_PI) {
                delta += (2 * CV_PI);
            }
            delta_sum += abs(delta);
        }

        return delta_sum - 2 * CV_PI < 1e-4;
    }

    /**
     * if the point in the quad
     * ref: http://alienryderflex.com/polygon/
     * @param p
     * @param vertexes
     * @return
     */
    inline bool pnpoly(const Point &quad_id, const Point &p) const {
        bool status = false;
        const int nums_vertex = 4;

        const int q_i = quad_id.y, q_j = quad_id.x;
        const auto &q1 = data.at<Point>(q_i, q_j);
        const auto &q2 = data.at<Point>(q_i, q_j + 1);
        const auto &q3 = data.at<Point>(q_i + 1, q_j);
        const auto &q4 = data.at<Point>(q_i + 1, q_j + 1);
        const Point *vertexes[] = {&q1, &q2, &q3, &q4};

        auto min_x = std::min(q1.x, q3.x);
        auto min_y = std::min(q1.y, q2.y);
        auto max_x = std::min(q2.x, q4.x);
        auto max_y = std::min(q3.y, q4.y);

        if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) {
            return false;
        }

        for (int i = 0, j = nums_vertex - 1; i < nums_vertex; j = i++) {
            if (((vertexes[i]->y > p.y) != (vertexes[j]->y > p.y)) &&
                (p.x < (vertexes[j]->x - vertexes[i]->x)
                       * (p.y - vertexes[i]->y)
                       / (double) (vertexes[j]->y - vertexes[i]->y)
                       + vertexes[i]->x))
                status = !status;
        }
        return status;
    }
};