#pragma once

#include <memory>
#include <iostream>

#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "boundary_segment.hpp"
#include "util.hpp"


using namespace cv;
using std::vector;

#define MAX_ENERGY (int)1e5

enum SEAM_FLAG {
    BOUNDARY = -1, // Boundary
    HORI_SEAM = -2,
    VERT_SEAM = -3
};

class SeamCarver {
public:
    typedef vector<int> seam_t;

    SeamCarver(const Mat &rgb_src, const Mat &mask) : rgb_src(rgb_src), src(Mat()),
                                                      rows(rgb_src.rows), cols(rgb_src.cols) {
        cvtColor(rgb_src, src, COLOR_RGB2GRAY);
        src.convertTo(src, CV_32S);
        // rgb_src.copyTo(src);

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (is_mask(mask.at<uchar>(i, j))) {
                    src.at<int>(i, j) = BOUNDARY;
                }
            }
        }
    }

    void local_warping(Mat &displacement) {
        SegmentBBox bbox;
        BoundarySegment segment(src);
        seam_t seam((size_t) std::max(src.rows, src.cols));
        Mat mask_energy = Mat::zeros(src.size(), CV_32S);

        // displacement matrix
        displacement.create(src.size(), CV_32SC2);
        // initialize the displacement matrix
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                displacement.at<Point>(i, j) = Point(j, i);
            }
        }

        Mat image = rgb_src.clone();

        // bbox.rect = Rect(0, 0, src.cols, src.rows);
        // bbox.shift = Point(1,0);

        while (segment.find_longest(bbox)) {
            compute_mask_energy(mask_energy);

            if (bbox.type() == HORI) { // hori seam
                find_horizontal_seam(src(bbox.rect), mask_energy(bbox.rect), seam);
                for (int j = bbox.x_min(); j < bbox.x_max(); j++) {
                    int seam_i = seam[j - bbox.x_min()];
                    auto seam_p = Point(j, seam_i);

                    if (bbox.shift.y > 0) { // seam -> bottom
                        for (int i = rows - 1; i > seam_i; i--) {
                            src.at<int>(i, j) = src.at<int>(i - 1, j);
                            displacement.at<Vec2i>(i, j) = displacement.at<Vec2i>(i - 1, j);
                            image.at<Vec3b>(i, j) = image.at<Vec3b>(i - 1, j);
                        }
                    } else { // seam -> top
                        for (int i = 0; i < seam_i; i++) {
                            src.at<int>(i, j) = src.at<int>(i + 1, j);
                            displacement.at<Vec2i>(i, j) = displacement.at<Vec2i>(i + 1, j);
                            image.at<Vec3b>(i, j) = image.at<Vec3b>(i + 1, j);
                        }
                    }

                    src.at<int>(seam_p) = (j % 2) ? HORI_SEAM : get_tb_avg(src, seam_p);
                    // src.at<int>(seam_p) = HORI_SEAM;
                    // src.at<int>(seam_p) = get_tb_avg(src, seam_p);

                    image.at<Vec3b>(seam_p) = Vec3b(0, 255, 0);
                }
            } else { // vert seam
                find_vertical_seam(src(bbox.rect), mask_energy(bbox.rect), seam);
                for (int i = bbox.y_min(); i < bbox.y_max(); i++) {
                    int seam_j = seam[i - bbox.y_min()];
                    auto seam_p = Point(seam_j, i);

                    if (bbox.shift.x > 0) { // seam -> right
                        for (int j = cols - 1; j > seam_j; j--) {
                            src.at<int>(i, j) = src.at<int>(i, j - 1);
                            displacement.at<Vec2i>(i, j) = displacement.at<Vec2i>(i, j - 1);
                            image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j - 1);
                        }
                    } else { // seam -> left
                        for (int j = 0; j < seam_j; j++) {
                            src.at<int>(i, j) = src.at<int>(i, j + 1);
                            displacement.at<Vec2i>(i, j) = displacement.at<Vec2i>(i, j + 1);
                            image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j + 1);
                        }
                    }

                    src.at<int>(seam_p) = (i % 2) ? VERT_SEAM : get_lr_avg(src, seam_p);
                    // src.at<int>(seam_p) = VERT_SEAM;
                    // src.at<int>(seam_p) = get_lr_avg(src, seam_p);

                    image.at<Vec3b>(seam_p) = Vec3b(255, 0, 0);
                }
            }

            // imshow("image", image);
            // waitKey(-1);
        }

        imshow("image", image);
        waitKey(-1);
        // std::cout << "local warping ok" << std::endl;
    }


private:
    const Mat &rgb_src;
    Mat src;

    int rows, cols;

    /**
     * compute the energy mask, using the magic value assigned to image source
     * @param mask_energy
     */
    void compute_mask_energy(Mat &mask_energy) const noexcept {
        mask_energy.setTo(0);

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (src.at<int>(i, j) <= BOUNDARY) {
                    mask_energy.at<int>(i, j) = MAX_ENERGY;
                }
            }
        }
    }

    /**
     * compute the cumulative energy map
     * @param energy
     * @param cumulative_energy
     * @param tracking_offset
     */
    void compute_cumulative_energy(const Mat &energy,
                                   Mat &cumulative_energy, Mat &tracking_offset) const noexcept {
        cumulative_energy = Mat::zeros(energy.size(), CV_32S);
        tracking_offset = Mat::zeros(energy.size(), CV_32S);

        energy(Range(0, 1), Range::all()).copyTo(cumulative_energy(Range(0, 1), Range::all()));

        for (int i = 1; i < energy.rows; i++) {
            for (int j = 0; j < energy.cols; j++) {
                auto _min = std::numeric_limits<int>::max();
                for (int k = -1; k < 2; k++) {
                    auto col = j + k;
                    if (col < 0 || col > energy.cols - 1) { continue; }

                    auto ce = cumulative_energy.at<int>(i - 1, col);
                    if (ce < _min) {
                        _min = ce;
                        tracking_offset.at<int>(i, j) = k;
                    }
                }
                cumulative_energy.at<int>(i, j) = _min + energy.at<int>(i, j);
            }
        }

        // std::cout << "cE: " << cumulative_energy << std::endl;
        // std::cout << "tracking: " << tracking_offset << std::endl;
    }

    /**
     * compute the forward energy on an image
     * ref: https://github.com/axu2/improved-seam-carving
     * @param src image source
     * @param mask_energy a user-defined energy mask
     * @param cumulative_energy
     * @param tracking_offset offset for the seam duration(left/top/right)
     */
    void compute_forward_energy(const Mat &src, const Mat &mask_energy,
                                Mat &cumulative_energy, Mat &tracking_offset) const noexcept {
        // auto energy = Mat::zeros(src.size(), CV_32S);
        cumulative_energy = Mat::zeros(src.size(), CV_32S);
        tracking_offset = Mat::zeros(src.size(), CV_32S);
        Mat energy = Mat::zeros(src.size(), CV_32S);

        Mat U, L, R;
        roll_row_shift(src, U, 1);
        roll_col_shift(src, L, 1);
        roll_col_shift(src, R, -1);

        Mat cU, cL, cR;
        absdiff(R, L, cU);
        absdiff(U, L, cL);
        cL += cU;
        absdiff(U, R, cR);
        cR += cU;

        for (int i = 1; i < src.rows; i++) {
            // m -> 1xcols matrix
            auto mU = cumulative_energy(Range(i - 1, i), Range::all());
            Mat mL, mR;
            roll_col_shift(mU, mL, 1);
            roll_col_shift(mU, mR, -1);

            auto distU = mU + cU(Range(i, i + 1), Range::all());
            auto distL = mL + cL(Range(i, i + 1), Range::all());
            auto distR = mR + cR(Range(i, i + 1), Range::all());

            Mat min_idx, min_val;
            triple_min<int>(distL, distU, distR, min_idx, min_val);

            // add additional pixel based energy measure
            min_val += mask_energy(Range(i, i + 1), Range::all());

            // [0, 1, 2] -> [-1, 0, 1]
            min_idx -= 1;

            auto _energy = min_val.clone();
            for (int j = 0; j < src.cols; j++) {
                int k = min_idx.at<int>(0, j);
                if (k == -1) {
                    _energy.at<int>(0, j) = cL.at<int>(i, j);
                } else if (k == 0) {
                    _energy.at<int>(0, j) = cU.at<int>(i, j);
                } else {
                    _energy.at<int>(0, j) = cR.at<int>(i, j);
                }
            }
            _energy.copyTo(energy(Range(i, i + 1), Range::all()));

            min_val.copyTo(cumulative_energy(Range(i, i + 1), Range::all()));
            min_idx.copyTo(tracking_offset(Range(i, i + 1), Range::all()));
        }

        energy += mask_energy;
        compute_cumulative_energy(energy, cumulative_energy, tracking_offset);

        // std::cout << "++cE: " << cumulative_energy << std::endl;
        // std::cout << "++tracking: " << tracking_offset << std::endl;
    }

    /**
     * find vertical seam
     * @param src image source
     * @param mask mark the stitched panorama area as `black`, else as `white`
     * @param seam the output seam
     */
    void find_vertical_seam(const Mat &src, const Mat &mask, seam_t &seam) const noexcept {
        Mat cumulative_energy, tracking_offset;
        // Mat cumulative_energy = Mat::zeros(src.size(), CV_32S);
        // Mat tracking_offset = Mat::zeros(src.size(), CV_32S);
        compute_forward_energy(src, mask, cumulative_energy, tracking_offset);

        // show_energy_map(cumulative_energy);
        // waitKey(-1);

        int min_energy = cumulative_energy.at<int>(src.rows - 1, 0);
        int min_index = 0;
        for (int j = 1; j < src.cols; j++) {
            auto e = cumulative_energy.at<int>(src.rows - 1, j);
            if (e < min_energy) {
                min_energy = e;
                min_index = j;
            }
        }
        seam[src.rows - 1] = min_index;

        for (int i = src.rows - 2; i >= 0; i--) {
            int j = seam[i + 1];
            auto offset = tracking_offset.at<int>(i, j);
            j += offset;
            seam[i] = j;
        }
    }

    /**
     * find horizontal seam via a transpose operation to vertical seam
     * @param src
     * @param mask
     * @param seam
     */
    void find_horizontal_seam(const Mat &src, const Mat &mask, seam_t &seam) const noexcept {
        auto _src = src.t();
        auto _mask = mask.t();
        find_vertical_seam(_src, _mask, seam);
    }

    /**
     * compute the left-right average value for one pixel
     * @param src
     * @param p
     * @return
     */
    inline int get_lr_avg(const Mat &src, const Point &p) {
        int i = p.y, j = p.x;
        int left, right;
        if (j > 0) {
            left = src.at<int>(i, j - 1);
        } else {
            // if the current pixel is left bound, assign the left as right
            left = src.at<int>(i, j + 1);
        }
        if (j < src.cols - 1) {
            right = src.at<int>(i, j + 1);
        } else {
            right = src.at<int>(i, j - 1);
        }
        auto ret = ((double) left + (double) right) / 2 + 0.5;
        return (int) ret;
    }

    /**
     * compute the top-bottom average value for one pixel
     * @param src
     * @param p
     * @return
     */
    inline int get_tb_avg(const Mat &src, const Point &p) {
        int i = p.y, j = p.x;
        int top, bottom;
        if (i > 0) {
            top = src.at<int>(i - 1, j);
        } else {
            top = src.at<int>(i + 1, j);
        }
        if (i < src.rows - 1) {
            bottom = src.at<int>(i + 1, j);
        } else {
            bottom = src.at<int>(i - 1);
        }
        auto ret = ((double) top + (double) bottom) / 2 + 0.5;
        return (int) ret;
    }

};