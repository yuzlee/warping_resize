#pragma once

#include <opencv2/core.hpp>

#include "../3rdparty/lsd-1.5/lsd.h"

using namespace cv;

class LsdImage {
public:
    typedef unsigned int uint_t;

    /**
     *
     * @param src input gray image, CV_64FC1
     */
    explicit LsdImage(const Mat &src) {
        cvtColor(src, cv_data, CV_RGB2GRAY);
        cv_data.convertTo(cv_data, CV_64FC1);

        origin = new image_double_s;

        origin->xsize = (uint_t) src.cols;
        origin->ysize = (uint_t) src.rows;

        origin->data = cv_data.ptr<double>(0);
    }

    inline image_double ptr() const {
        return origin;
    }

private:
    image_double origin = nullptr;
    Mat cv_data;
};

struct LsdLine {
    double x1;
    double y1;
    double x2;
    double y2;
    double width;

    inline Point start() {
        return Point((int) lround(x1), (int) lround(y1));
    }

    inline Point end() {
        return Point((int) lround(x2), (int) lround(y2));
    }
};

class LsdLineList {
public:
    LsdLineList() = default;

    inline void from(ntuple_list_s *origin) {
        this->origin = origin;
    }

    inline LsdLine &operator[](int index) {
        auto *data = (LsdLine *) &origin->values[index * origin->dim];
        return *data;
    }

    inline int length() const {
        if (origin == nullptr) {
            return 0;
        }
        return origin->size;
    }

private:
    ntuple_list_s *origin = nullptr;
};

static void detect_line_segment(const Mat &src, LsdLineList &list, double scale = 0.8) {
    auto img = LsdImage(src);
    auto *t = lsd_scale(img.ptr(), scale);
    list.from(t);
}