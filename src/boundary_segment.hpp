#pragma once

#include <vector>
#include <opencv2/core.hpp>

using namespace cv;
using std::vector;


enum Direction {
    VERT,
    HORI
};

// description for segments in one side
struct SegmentMeta {
    Point from;
    Point to;

    inline int length() {
        return abs(from.x - to.x) + abs(from.y - to.y);
    }
};

struct SegmentBBox {
    Rect2i rect;
    Point shift;

    inline int x_min() { return rect.x; }
    inline int y_min() { return rect.y; }
    inline int x_max() { return rect.x + rect.width; }
    inline int y_max() { return rect.y + rect.height; }

    inline int type() { return (shift.x == 0) ? HORI : VERT; }
};

class BoundarySegment {
public:
    explicit BoundarySegment(const Mat& src): src(src), rows(src.rows), cols(src.cols) {

    }

    bool find_longest(SegmentBBox& bbox) {
        vector<SegmentMeta> meta;

        // left: top<->bottom
        find_boundary_segment(Point(0, 0), Point(0, rows-1), meta);
        // right: top<->bottom
        find_boundary_segment(Point(cols-1, 0), Point(cols-1, rows-1), meta);
        // top: left<->right
        find_boundary_segment(Point(0, 0), Point(cols-1, 0), meta);
        // bottom: left<->right
        find_boundary_segment(Point(0, rows-1), Point(cols-1, rows-1), meta);

        vector<SegmentMeta>::iterator max_iter;
        int max_length = 0;
        for (auto it = meta.begin(); it != meta.end(); it++) {
            int length = it->length();
            if (length > max_length) {
                max_length = length;
                max_iter = it;
            }
        }

        if (max_length < 1) {
            return false;
        }

        Point& from = max_iter->from;
        Point& to = max_iter->to;

        if (from.x == to.x) { // vert
            bbox.rect = Rect2i(0, from.y, cols, (to.y - from.y + 1));
            int u_x = from.x == 0 ? -1 : 1;
            bbox.shift = Point(u_x, 0);
        } else if (from.y == to.y) { // hori
            bbox.rect = Rect2i(from.x, 0, (to.x - from.x + 1), rows);
            int u_y = from.y == 0 ? -1 : 1;
            bbox.shift = Point(0, u_y);
        } else {
            return false;
        }
        return (bbox.rect.width > 1 && bbox.rect.height > 1);
    }

private:
    const Mat& src;
    int rows, cols;

    inline bool is_blank_pixel(int x) {
        return x  == -1;
    }

    inline bool is_blank_pixel(const Point& p) {
        return is_blank_pixel(src.at<int>(p));
    }

    inline bool is_blank_pixel(int i, int j) {
        return is_blank_pixel(src.at<int>(i, j));
    }

    inline bool is_blank_pixel_checked(int i, int j) {
        if (i >= 0 && i < rows && j >= 0 && j < cols) {
            return is_blank_pixel(i, j);
        }
        return false;
    }

    inline void find_boundary_segment(const Point& from, const Point& to, vector<SegmentMeta>& meta) {
        Point delta;
        if (from.x == to.x) { // vert
            if (from.x <= to.x) {
                delta = Point(0, 1);
            } else {
                delta = Point(0, -1);
            }
        } else if (from.y == to.y) { // hori
            if (from.x <= to.x) {
                delta = Point(1, 0);
            } else {
                delta = Point(-1, 0);
            }
        } else {
            return;
        }

        Point cursor = from;
        bool in_segment = false;
        for (; cursor.x <= to.x && cursor.y <= to.y; cursor += delta) {
            if (is_blank_pixel(cursor)) {
                if (!in_segment) {
                    meta.emplace_back(SegmentMeta());
                    meta.rbegin()->from = cursor;
                    in_segment = true;
                }
            } else if (in_segment) {
                meta.rbegin()->to = cursor;
                in_segment = false;
            }
        }
        if (!meta.empty()) {
            meta.rbegin()->to = to;
        }
    }
};
