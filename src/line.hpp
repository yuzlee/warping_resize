#pragma once

#include "interpolation.hpp"
#include "util.hpp"

using namespace cv;

class Line {
public:
    const Point start;
    const Point end;
    double theta = 0; // relative rotation angle

    Line(const Point &s, const Point &e) : start(s), end(e), theta(0) {}

    /**
     * quantize the line orientation into M = 50 bins
     * @return bin index
     */
    inline const int group_id(const int M = 50) const {
        auto angle = orientation_angle();
        return (int) lround((angle + CV_PI / 2) / (CV_PI / (M - 1))); // [0,49]
    }

    inline const Vec2d difference() const {
        auto diff = start - end;
        return Vec2d(diff.x, diff.y);
    }

    inline const Matx28d translation_difference() const {
        return T_start - T_end;
    }

    inline const Vec2d start_d() const {
        return Vec2d(start.x, start.y);
    }

    inline const Vec2d end_d() const {
        return Vec2d(end.x, end.y);
    }

    /**
     *
     * @param V_q the vertexes in a quad, 8x1 int vector
     * @return
     */
    inline double get_rotation_angle(const Matx81d &V_q) {
        assert(V_q.rows == 8 && V_q.cols == 1);

        Matx21d _s = (T_start * V_q);
        Matx21d _e = (T_end * V_q);
        Point s((int) lround(_s(0, 0)), (int) lround(_s(1, 0)));
        Point e((int) lround(_e(0, 0)), (int) lround(_e(1, 0)));
        Line rhs{s, e};
        auto delta = rhs.orientation_angle() - this->orientation_angle();
        if (delta > CV_PI / 2) {
            delta -= CV_PI;
        }
        if (delta < -CV_PI / 2) {
            delta += CV_PI;
        }

        return delta;
    }

    /**
     * compute translation matrix from original quad
     * @param quad 2x2 quad, Vec2i in each element
     */
    inline void compute_translation_matrix(const Matx81d &V_q) {
        assert(V_q.rows == 8 && V_q.cols == 1);

        // TODO: prove the equation
        // p = T * V, T = p * V.t() * (V * V.t()).inv()
        // Matx18d V_term = V_q.t() * (V_q * V_q.t() + 0.01 * Matx88d::eye()).inv();
        // T_start = startd() * V_term;
        // T_end = endd() * V_term;
        T_start = compute_translation(V_q, start_d());
        T_end = compute_translation(V_q, end_d());
    }

    /**
     * compute the intersection point between this line and another line
     * ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
     * @param s another line start point
     * @param e another line end point
     * @param p[out] intersection point
     * @param param_t[out] the `t` parameter
     * @return if the lines have a intersection point
     */
    bool line_intersection(const Point &s, const Point &e, Point &p) {
        int denominator = (start.x - end.x) * (s.y - e.y) - (start.y - end.y) * (s.x - e.x);
        if (denominator == 0) { // parallel or coincident
            return false;
        }
        int _t = (start.x - s.x) * (s.y - e.y) - (start.y - s.y) * (s.x - e.x);
        int _u = (start.y - end.y) * (start.x - s.x) - (start.x - end.x) * (start.y - s.y);
        auto t = (double) _t / denominator;
        auto u = (double) _u / denominator;

        if (0 <= u && u <= 1 && 0 <= t && t <= 1) {
            p.x = (int) lround(start.x + t * (end.x - start.x));
            p.y = (int) lround(start.y + t * (end.y - start.y));
            return true;
        }
        return false;
    }

    static inline Line init_line(const Point &s, const Point &e, const Matx81d &quad) {
        Line line{s, e};
        line.compute_translation_matrix(quad);
        return line;
    };

private:
    Matx28d T_start; // 2x8 matrix
    Matx28d T_end;

    /**
     * absolute rotation angle (orientation)
     * @return
     */
    inline const double orientation_angle() const {
        auto angle = atan2((double) (start.y - end.y), (double) (start.x - end.x));
        if (angle < -CV_PI / 2) {
            angle += CV_PI;
        }
        if (angle >= CV_PI / 2) {
            angle -= CV_PI;
        }
        return angle;
    }
};