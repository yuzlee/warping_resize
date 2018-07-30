#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "lsd.hpp"
#include "util.hpp"

#include "line.hpp"
#include "mesh.hpp"
#include "optimizer.hpp"

#include "interpolation.hpp"
#include "matrix_data.hpp"

using namespace cv;
using namespace Eigen;
using std::map;
using std::vector;

struct OptimizationOption {
    int nums_iteration; // nums_iteration

    int mesh_cols; // vertex
    int mesh_rows;

    double lambda_l; // lambda_line
    double lambda_b; // lambda_boundary

    static inline OptimizationOption get_default() {
        return {10, 10, 5, 100, 1e8};
    }
};


class EnergyOptimizer {
public:
    EnergyOptimizer(const Mat &src, const Mat &mask) : src(src), mask(mask), solver(new QRSolver()) {
    }

    ~EnergyOptimizer() {
        delete solver;
    }

    /**
     *
     * @param disp CV_32SC2, d_ij = (x, y), where (x,y) is the pixel of the original image
     * @param lambda_l
     * @param lambda_b
     */
    void global_warping(const Mat &disp, const OptimizationOption &option) {
        Size mesh_size(option.mesh_cols, option.mesh_rows);

        // warp the rectangle mesh using displacement filed
        mesh.init_from_image(src.size(), mesh_size).displace(disp);

        // translate the warp to 1-based coordinate
        mesh.offset();

        // optimization
        opt_mesh.init(mesh_size);

        auto shape_optimizer = ShapeOptimizer(mesh);
        auto line_optimizer = LineOptimizer<50>(mesh);
        // find and warp the original line segment using displacement filed
        line_optimizer.find_and_init_lines(src);

        // size of shape-line-energy matrix
        const auto K_rows = 8 * mesh.quad_rows * mesh.quad_cols + 2 * line_optimizer.get_nums_line();
        const auto K_cols = 8 * mesh.quad_rows * mesh.quad_cols;

        const SMatrixd K(K_rows, K_cols);
        const double lambda_s = 1;
        shape_optimizer.make_block_diag(const_cast<SMatrixd &>(K), lambda_s, 0, 0);
        const_cast<SMatrixd &>(K).makeCompressed();

        // weight matrix for shape-line-matrix
        const SMatrixd Q = mesh.Q();

        // boundary matrix
        const auto nums_vertex = 2 * mesh.vertex_rows * mesh.vertex_cols;
        const SMatrixd B_coeffs(nums_vertex, nums_vertex);
        const SVectord b_term(nums_vertex + K_rows, 1);
        compute_boundary_matrix(const_cast<SMatrixd &>(B_coeffs), const_cast<SVectord &>(b_term), option.lambda_b);

        /*
         * A = [ B_coeffs; K ]                  (2*V_NUMS+(8+2*L)*Q_NUMS)x(2*V_NUMS)
         * V = [ v1_x; v1_y; ...; vn_x; vn_y ]  (2*V_NUMS)x1
         * b_term = [ Boundary; 0]              (2*V_NUMS+(8+2*L)*Q_NUMS)x1
         */

        for (int it = 0; it < option.nums_iteration; it++) {
            // fix theta, update V
            auto K_copy = SMatrixd(K); // a deep copy
            const int K_row_offset = K_cols; // Shape Matrix is K_cols x K_cols
            line_optimizer.make_block_diag(K_copy, option.lambda_l, K_row_offset, 0);
            auto K_weighted = SMatrixd(K_copy * Q);

            SMatrixd A = eigen_row_concat(B_coeffs, K_weighted);

            // std::cout << "A" << std::endl << K_copy << std::endl;
            // std::cout << "b_term" << std::endl << b_term << std::endl;
            // const string root = "./data/" + std::to_string(it) + "-";
            // write_csv(root + "A.csv", A);
            // write_csv(root + "b.csv", SMatrixd(b_term));

            this->solve_mesh(A, b_term);

            // opt_mesh.show_mesh(src, "mesh - it = " + std::to_string(it));

            // fix V, update theta
            line_optimizer.update_line_theta(opt_mesh);

            std::cout << "iteration " << it << " - ok" << std::endl;
        }

        // opt_mesh.show_mesh(src, "mesh - end");
    }

    void generate(Mat &target) {
        // interpolate_displacement(target);
        warp_mesh(src, mesh, opt_mesh, target);

        // reduce_stretching_distortion(target);
    }

private:
    const Mat &src;
    const Mat &mask;

    /**
     * Mesh Map, CV_32SU2
     */
    Mesh mesh;

    Mesh opt_mesh;

    bool pattern_analyzed = false;
    QRSolver *solver = nullptr;

    /**
     *
     * @param shape MxNx8x8 tensor
     * @param line MxNx(Lx2)x8 tensor
     */
    void solve_mesh(const SMatrixd &A, const SVectord &b) {
        // TODO: fix theta, solving linear system

        // A * V = b
        // V = inv(A'A + lambda*I)A'b
        // ridge regression
        // const auto A_t = SMatrixd(A.transpose());
        // SMatrixd V = eigen_inv(A_t * A) * A_t * b;

        // if (!pattern_analyzed) {
        //     solver->analyzePattern(A);
        //     pattern_analyzed = true;
        // }
        // solver->factorize(A);
        solver->compute(A);
        SMatrixd V = solver->solve(b);

        // std::cout << "computed V:" << std::endl << V << std::endl;

        for (int i = 0; i < mesh.vertex_rows; i++) {
            for (int j = 0; j < mesh.vertex_cols; j++) {
                int k = (i * mesh.vertex_cols + j) * 2;
                auto x = (int) lround(V.coeff(k, 0));
                auto y = (int) lround(V.coeff(k + 1, 0));
                opt_mesh.at(i, j) = {x, y};
            }
        }
    }

    /**
     * bilinearly interpolate the displacement of any pixel
     * from the displacement of the four quad vertexes
     * and generate the target image
     * @param target_disp
     */
    void interpolate_displacement(Mat &target) {
        target.create(src.size(), src.type());

        Mat disp;
        Mat delta = opt_mesh.mat() - mesh.mat();

        assert(delta.channels() == 2);

        delta.convertTo(delta, CV_32FC2);

        resize(delta, disp, src.size());

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                auto d = disp.at<Point>(i, j);
                auto dx = (int) lround(d.x);
                auto dy = (int) lround(d.y);
                target.at<Vec3b>(i, j) = src.at<Vec3b>(i + dy, j + dx);
            }
        }
    }

    /**
     * simply resize the image
     * @param target[in,out] target image
     */
    void reduce_stretching_distortion(Mat &target) {
        double s_x = 0, s_y = 0;
        for (int i = 0; i < mesh.quad_rows; i++) {
            for (int j = 0; j < mesh.quad_cols; j++) {
                const auto &p_min = opt_mesh.at(i, j);
                const auto &p_max = opt_mesh.at(i + 1, j + 1);
                const auto &p_min_hat = mesh.at(i, j);
                const auto &p_max_hat = mesh.at(i + 1, j + 1);

                s_x += (double) (p_max.x - p_min.x) / (p_max_hat.x - p_min_hat.x);
                s_y += (double) (p_max.y - p_min.y) / (p_max_hat.y - p_min_hat.y);
            }
        }
        const int n = mesh.quad_cols * mesh.quad_rows;
        s_x /= n;
        s_y /= n;

        resize(target, target, Size(), s_x, s_y);
    }

    // ********** Boundary Constraints *******************

    inline void compute_boundary_matrix(SMatrixd &B_coeffs, SVectord &b_term, double lambda) const {
        int n = 2 * mesh.vertex_rows * mesh.vertex_cols;
        int xn = mesh.vertex_cols, yn = mesh.vertex_rows;

        for (int k = 0; k < n; k += 2 * xn) { // left boundary `x`
            B_coeffs.insert(k, k) = 1;
            b_term.insert(k) = 1;
        }

        for (int k = 2 * xn - 2; k < n; k += 2 * xn) { // right boundary `x`
            B_coeffs.insert(k, k) = 1;
            b_term.insert(k) = (src.cols);
        }

        for (int k = 1; k < 2 * xn; k += 2) { // top boundary `y`
            B_coeffs.insert(k, k) = 1;
            b_term.insert(k) = 1;
        }

        for (int k = n - 2 * xn + 1; k < n; k += 2) { // bottom boundary `y`
            B_coeffs.insert(k, k) = 1;
            b_term.insert(k) = (src.rows);
        }

        // std::cout << "Boundary Coeffs:" << std::endl << B_coeffs << std::endl;
        // std::cout << "b_term:" << std::endl << b_term.topRows(n) << std::endl;

        B_coeffs *= lambda;
        b_term *= lambda;
    }

};