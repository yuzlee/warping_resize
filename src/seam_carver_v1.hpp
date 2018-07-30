#pragma once

#include <memory>
#include <iostream>

#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "boundary_segment.hpp"

using namespace cv;
using std::vector;

#define MAX_ENERGY (int)1e5

class SeamCarverV1 {
public:
	typedef vector<int> seam_t;
	typedef int dist_t;

	SeamCarverV1(Mat& src, const Mat& mask)
		: src(src), rows(src.rows), cols(src.cols), mask(mask),
		energy(src.size(), CV_32S), dist(src.size(), CV_32S), edge(src.size(), CV_32S)
    {
		
    }

	void local_warping() {
		SegmentBBox bbox;
		seam_t seam((size_t)std::max(rows, cols));

		Mat shift_mat = Mat::zeros(src.size(), CV_32SC2);
		Mat src_cp = src.clone();

		compute_energy(Rect(0, 0, src.cols, src.rows));

		while (find_longest_boundary_segment(bbox)) {
			if (bbox.type() == HORI) { // hori seam
				find_horizontal_seam(bbox.rect, seam);
				for (int j = bbox.x_min(); j < bbox.x_max(); j++) {
					int seam_i = seam[j];
					auto seam_p = Point(j, seam_i);

					if (bbox.shift.y > 0) { // seam -> bottom
						for (int i = rows - 1; i > seam_i; i--) {
							src.at<Vec3b>(i, j) = src.at<Vec3b>(i - 1, j);
							energy.at<dist_t>(i, j) = energy.at<dist_t>(i - 1, j);
							mask.at<uchar>(i, j) = mask.at<uchar>(i - 1, j);
						}
					} else { // seam -> top
						for (int i = 0; i < seam_i; i++) {
							src.at<Vec3b>(i, j) = src.at<Vec3b>(i + 1, j);
							energy.at<dist_t>(i, j) = energy.at<dist_t>(i + 1, j);
							mask.at<uchar>(i, j) = mask.at<uchar>(i + 1, j);
						}
					}

					src.at<Vec3b>(seam_p) = Vec3b(0, 0, 255);
					energy.at<dist_t>(seam_p) = MAX_ENERGY;
				}
			} else { // vert seam
				find_vertical_seam(bbox.rect, seam);
				for (int i = bbox.y_min(); i < bbox.y_max(); i++) {
					int seam_j = seam[i];
					auto seam_p = Point(seam_j, i);

					if (bbox.shift.x > 0) { // seam -> right
						for (int j = cols - 1; j > seam_j; j--) {
							src.at<Vec3b>(i, j) = src.at<Vec3b>(i, j - 1);
							energy.at<dist_t>(i, j) = energy.at<dist_t>(i, j - 1);
							mask.at<uchar>(i, j) = mask.at<uchar>(i, j - 1);
						}
					} else { // seam -> left
						for (int j = 0; j < seam_j; j++) {
							src.at<Vec3b>(i, j) = src.at<Vec3b>(i, j + 1);
							mask.at<uchar>(i, j) = mask.at<uchar>(i, j + 1);
						}
					}

					src.at<Vec3b>(seam_p) = Vec3b(0,255,0);
					energy.at<dist_t>(seam_p) = MAX_ENERGY;
				}
			}

			imshow("image", src);
			waitKey(-1);
		}

		imshow("image", src);
		waitKey(-1);
	}

    void show() {
        SegmentBBox bbox;
        auto status = find_longest_boundary_segment(bbox);
		// auto status = true;
		// bbox.rect = Rect2i(0, 0, src.cols, src.rows);
		// bbox.shift = Point(1, 1);

        Mat dest = src.clone();
        if (status) {
			seam_t seam(std::max(rows, cols));

			std::cout << "shift: " << bbox.shift << std::endl;
			std::cout << "bbox: " << bbox.rect << std::endl;

            if (bbox.type() == HORI) { // hori seam
				std::cout << "HORI" << std::endl;

                find_horizontal_seam(bbox.rect, seam);
                for (int j = bbox.x_min(); j < bbox.x_max(); j++) {
                    int i = seam[j];

					// std::cout << Point(j, i) << std::endl;

					// circle(dest, Point(j, i), 2, Vec3b(255, 0, 0));
                    dest.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
                }
            } else { // vert seam
				std::cout << "VERT" << std::endl;

                find_vertical_seam(bbox.rect, seam);
                for (int i = bbox.y_min(); i < bbox.y_max(); i++) {
                    int j = seam[i];
					// std::cout << Point(j, i) << std::endl;
					// circle(dest, Point(j, i), 2, Vec3b(255, 0, 0));
                    dest.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
                }
			}

			rectangle(dest, bbox.rect, Scalar(255, 255, 0), 1, 8, 0);

			imshow("seam", dest);

			Mat energy_map(src.size(), CV_8UC3);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					auto e = energy.at<dist_t>(i, j);
					if (e == MAX_ENERGY) {
						energy_map.at<Vec3b>(i, j) = Vec3b(0,255,0);
					} else {
						e = (int)(log(e+1) * 21);
						energy_map.at<Vec3b>(i, j) = Vec3b(e, e, e);
					}
					// energy_map.at<uchar>(i, j) = (uchar)e;
				}
			}
			imshow("energy map", energy_map);

			waitKey(-1);
        }
    }

    void find_vertical_seam(const Rect& rect, seam_t& seam) {
		int row_0 = rect.y, col_0 = rect.x,
			row_n = row_0 + rect.height,
			col_n = col_0 + rect.width;

        compute_energy(rect);

		// seam.resize(rows);
        dist.setTo(std::numeric_limits<int>::max());
        edge.setTo(0);

        // initialize the distance, set top to 0
        for (int j = col_0; j < col_n; j++) {
			dist.at<dist_t>(row_0, j) = 0; // energy.at<dist_t>(row_0, j);
        }

		for (int i = row_0 + 1; i < row_n; i++) {
			dist.at<dist_t>(i, col_0) = energy.at<dist_t>(i, col_0) + dist.at<dist_t>(i - 1, col_0);
			edge.at<int>(i, col_0) = 0;

			dist.at<dist_t>(i, col_n - 1) = energy.at<dist_t>(i, col_n - 1) + dist.at<dist_t>(i - 1, col_n - 1);
			edge.at<int>(i, col_n - 1) = 0;

			for (int j = col_0 + 1; j < col_n - 1; j++) {
				dist_t e = energy.at<dist_t>(i, j);
				// if (e == MAX_ENERGY) { continue; }
				dist_t& current_dist = dist.at<dist_t>(i, j);
				int& edge_to = edge.at<int>(i, j);

				dist_t tl = dist.at<dist_t>(i - 1, j - 1);
				dist_t tr = dist.at<dist_t>(i - 1, j + 1);
				dist_t t = dist.at<dist_t>(i - 1, j);

				triple_min(tl, t, tr, e, current_dist, edge_to);
			}
		}

        // find min energy at bottom side
        int min_index = 0, min = dist.at<dist_t>(row_n - 1, col_0);
        for (int j = col_0 + 1; j < col_n; j++) {
            int _dist = dist.at<dist_t>(row_n - 1, j);
            if (_dist < min) {
                min_index = j;
                min = _dist;
            }
        }

		std::cout << "min index: " << min_index << ", min: " << min << std::endl;

        // backtracing
        seam[row_n - 1] = min_index;
        for (int i = row_n - 1; i > row_0; i--) {
            seam[i - 1] = seam[i] + edge.at<int>(i, seam[i]);
        }
    }

    void find_horizontal_seam(const Rect& rect, seam_t& seam) {
        int row_0 = rect.y, col_0 = rect.x,
            row_n = row_0 + rect.height,
            col_n = col_0 + rect.width;

        compute_energy(rect);

        // seam.resize(cols);
        dist.setTo(std::numeric_limits<int>::max());
        edge.setTo(0);

        // initialize the distance, set left to 0
        for (int i = row_0; i < row_n; i++) {
			dist.at<dist_t>(i, col_0) = 0; // energy.at<dist_t>(i, col_0);
        }

		for (int j = col_0 + 1; j < col_n; j++) {
			dist.at<dist_t>(row_0, j) = energy.at<dist_t>(row_0, j) + dist.at<dist_t>(row_0, j - 1);
			edge.at<int>(row_0, j) = 0;

			dist.at<dist_t>(row_n - 1, j) = energy.at<dist_t>(row_n - 1, j) + dist.at<dist_t>(row_n - 1, j - 1);
			edge.at<int>(row_n - 1, j) = 0;

			for (int i = row_0 + 1; i < row_n - 1; i++) {
				dist_t e = energy.at<dist_t>(i, j);
				// if (e == MAX_ENERGY) { continue; }
				dist_t& current_dist = dist.at<dist_t>(i, j);
				int& edge_to = edge.at<int>(i, j);

				dist_t tl = dist.at<dist_t>(i - 1, j - 1);
				dist_t bl = dist.at<dist_t>(i + 1, j - 1);
				dist_t l = dist.at<dist_t>(i, j - 1);

				triple_min(tl, l, bl, e, current_dist, edge_to);
			}
		}

        // find min energy at bottom side
        int min_index = 0, min = dist.at<dist_t>(row_0, col_n - 1);
        for (int i = row_0 + 1; i < row_n; i++) {
            int _dist = dist.at<dist_t>(i, col_n - 1);
            if (_dist < min) {
                min_index = i;
                min = _dist;
            }
        }

		std::cout << "min index: " << min_index << ", min: " << min << std::endl;

        // backtracing
        seam[col_n - 1] = min_index;
        for (int j = col_n - 1; j > col_0; j--) {
            seam[j - 1] = seam[j] + edge.at<int>(seam[j], j);
        }
    }

private:
    Mat& src;
	Mat mask;

    Mat energy;
    Mat dist;
    Mat edge;

    int rows;
    int cols;

	inline void triple_min(dist_t a, dist_t b, dist_t c, 
							dist_t e, dist_t& target, int& edge_flag) {
		if (a < b) {
			if (a < c) {
				target = e + a;
				edge_flag = -1;
			} else {  // a >= c
				target = e + c;
				edge_flag = 1;
			}
		} else {
			if (b < c) {
				target = e + b;
				edge_flag = 0;
			} else {
				target = e + c;
				edge_flag = 1;
			}
		}
	}

	inline dist_t compute_energy(int i, int j) {
		dist_t e;
		if (is_blank_pixel(i, j)) {
			e = MAX_ENERGY;
		} else {
			auto diff_lr = (Vec3i)src.at<Vec3b>(i, j - 1) - (Vec3i)src.at<Vec3b>(i, j + 1);
			auto diff_tb = (Vec3i)src.at<Vec3b>(i - 1, j) - (Vec3i)src.at<Vec3b>(i + 1, j);
			e = diff_lr.dot(diff_lr) + diff_tb.dot(diff_tb);

			e = (int)sqrt(e);
		}
		return e;
	}

    void compute_energy(const Rect& rect) {
        int row_0 = rect.y,
            col_0 = rect.x,
            row_n = row_0 + rect.height,
            col_n = col_0 + rect.width;

		// mask.setTo(0);
		energy.setTo(MAX_ENERGY);

		// init mask for white boundary
		//for (int i = row_0; i < row_n; i++) {
		//	for (int j = col_0; j < col_n; j++) {
		//		if (is_blank_pixel(i, j)) {
		//			int flag = 3;
		//			// flag += is_blank_pixel_checked(i - 1, j);
		//			// flag += is_blank_pixel_checked(i + 1, j);
		//			// flag += is_blank_pixel_checked(i, j - 1);
		//			// flag += is_blank_pixel_checked(i, j + 1);
		//			if (flag >= 3) { // at least has 3 blank neighbors
		//				mask.at<uchar>(i, j) = 1;
		//			} else {
		//				mask.at<uchar>(i, j) = 0;
		//			}
		//		}
		//	}
		//}

        for (int i = row_0 + 1; i < row_n - 1; i++) {
            for (int j = col_0 + 1; j < col_n - 1; j++) {
				if (is_blank_pixel(i, j)) {
					energy.at<dist_t>(i, j) = MAX_ENERGY;
				} else {
					auto diff_lr = (Vec3i)src.at<Vec3b>(i, j - 1) - (Vec3i)src.at<Vec3b>(i, j + 1);
					auto diff_tb = (Vec3i)src.at<Vec3b>(i - 1, j) - (Vec3i)src.at<Vec3b>(i + 1, j);
					auto e = diff_lr.dot(diff_lr) + diff_tb.dot(diff_tb);

					energy.at<dist_t>(i, j) = (int)sqrt(e);
				}
            }
        }
    }

    // ************* find boundary segment ****************

    inline bool is_blank_pixel(const Vec3b& x) {
		const uchar t = 252;
        // return x[0] > t && x[1] > t && x[2] > t;
		return x[0] + x[1] + x[2] > t * 3;
    }

	inline bool is_blank_pixel(const uchar& x) {
		return x > 128;
	}

	inline bool is_blank_pixel(const Point& p) {
		return is_blank_pixel(mask.at<uchar>(p));
	}

	inline bool is_blank_pixel(int i, int j) {
		return is_blank_pixel(mask.at<uchar>(i, j));
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
					meta.push_back(SegmentMeta());
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

    bool find_longest_boundary_segment(SegmentBBox& bbox) {
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

			// line(dest, it->from, it->to, Scalar(rng.next()%255, rng.next() % 255, rng.next() % 255));
			// std::cout << "from: " << it->from << ", to: " << it->to << std::endl;
		}

		if (max_length < 5) {
			return false;
		}

        Point& from = max_iter->from;
        Point& to = max_iter->to;

		// std::cout << "from: " << from << ", to: " << to << std::endl;

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
        return (bbox.rect.width > 1 || bbox.rect.height > 1);
    }
};
