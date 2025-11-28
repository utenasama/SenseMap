#include "poisson_blending.h"

namespace sensemap {

typedef Eigen::Triplet<double> Triplet;

PoissonBlender::PoissonBlender() {}

PoissonBlender::PoissonBlender(const cv::Mat & mosaic, const cv::Mat & mask, const cv::Mat & drvxy) 
    : mosaic_(mosaic),
      mask_(mask),
      drvxy_(drvxy) {}

void PoissonBlender::Solve() {
    Eigen::SparseMatrix<double> A;
    Eigen::Matrix<double, Eigen::Dynamic, 1> b, u;

    std::cout << "Build Sparse Linear Matrix" << std::endl;
    BuildMatrix(A, b, u);

    std::cout << "Solve Poisson Equation" << std::endl;
    PoissonSolve(A, b, u);

    CopyResult(u);
}

bool PoissonBlender::BuildMatrix(Eigen::SparseMatrix<double> &A, Eigen::Matrix<double, Eigen::Dynamic, 1> &b,
                                 Eigen::Matrix<double, Eigen::Dynamic, 1> &u) {
    const int width = mosaic_.cols;
    const int height = mosaic_.rows;
    const int ch = 3;

    cv::Mat fimage;
    mosaic_.convertTo(fimage, CV_32FC4, 1.0 / 255.f, 0);

    int nz = 0;
    for (int y = 0; y < height; ++y) {
        uchar *p = mask_.ptr(y);
        for (int x = 0; x < width; ++x, ++p) {
            if ((*p) == 0) continue;
            int id = (y * width + x) * ch;
            mp_[id] = nz++;   // r
            mp_[++id] = nz++; // g
            mp_[++id] = nz++; // b
        }
    }

    A = Eigen::SparseMatrix<double>(nz, nz);
    b = Eigen::VectorXd(nz);
    u = Eigen::VectorXd(nz);

    std::cout << "variables: " << nz << std::endl;

    // A.reserve(5 * nz);
    // int rowA = 0;

    std::vector<Triplet> triplets;
    triplets.reserve(5 * nz);
    for (int y = 0; y < height; ++y) {
        uchar *p = mask_.ptr(y);
        cv::Vec3f *drv = drvxy_.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x, ++p, ++drv) {
            if (*p == 0) continue;

            int id = (y * width + x) * ch;
            int tidx = id - ch * width, lidx = id - ch, ridx = id + ch, bidx = id + ch * width;
            
            // To optimize insertion.
            unsigned char tlrb = 15; // 0b1111
            if (y > 0 && mask_.at<uchar>(y - 1, x) == 0) {
                cv::Vec4f rgba = fimage.at<cv::Vec4f>(y - 1, x);
                (*drv)[0] -= rgba[0];
                (*drv)[1] -= rgba[1];
                (*drv)[2] -= rgba[2];
                tlrb &= 7; // 0b0111
            } else if (y == 0) {
                tlrb &= 7; // 0b0111
            }
            if (x > 0 && mask_.at<uchar>(y, x - 1) == 0) {
                cv::Vec4f rgba = fimage.at<cv::Vec4f>(y, x - 1);
                (*drv)[0] -= rgba[0];
                (*drv)[1] -= rgba[1];
                (*drv)[2] -= rgba[2];
                tlrb &= 11; // 0b1011
            } else if (x == 0) {
                tlrb &= 11; // 0b1011
            }
            if (x < width - 1 && mask_.at<uchar>(y, x + 1) == 0) {
                cv::Vec4f rgba = fimage.at<cv::Vec4f>(y, x + 1);
                (*drv)[0] -= rgba[0];
                (*drv)[1] -= rgba[1];
                (*drv)[2] -= rgba[2];
                tlrb &= 13; // 0b1101
            } else if (x == width - 1) {
                tlrb &= 13; // 0b1101
            }
            if (y < height - 1 && mask_.at<uchar>(y + 1, x) == 0) {
                cv::Vec4f rgba = fimage.at<cv::Vec4f>(y + 1, x);
                (*drv)[0] -= rgba[0];
                (*drv)[1] -= rgba[1];
                (*drv)[2] -= rgba[2];
                tlrb &= 14; // 0b1110
            } else if (y == height - 1) {
                tlrb &= 14; // 0b1110
            }

            for (int k = 0; k < ch; ++k) {
                int cid = mp_.at(id + k);
                if (tlrb & 8) triplets.push_back(Triplet(cid, mp_.at(tidx + k), 1.0));
                if (tlrb & 4) triplets.push_back(Triplet(cid, mp_.at(lidx + k), 1.0));
                              triplets.push_back(Triplet(cid, cid,             -4.0));
                if (tlrb & 2) triplets.push_back(Triplet(cid, mp_.at(ridx + k), 1.0));
                if (tlrb & 1) triplets.push_back(Triplet(cid, mp_.at(bidx + k), 1.0));
                
                b(cid) = cv::saturate_cast<double>((*drv)[k]);
            }
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());

    return true;
}

bool PoissonBlender::PoissonSolve(const Eigen::SparseMatrix<double> &A, const Eigen::Matrix<double, Eigen::Dynamic, 1> &b, 
                                  Eigen::Matrix<double, Eigen::Dynamic, 1> &u) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > lu_of_A(A);
   
    lu_of_A.analyzePattern(A);
    std::cout << lu_of_A.lastErrorMessage() << std::endl;
    // if (lu_of_A.info() != Eigen::ComputationInfo::Success) {
    //     std::cerr<< "analyzePattern failed" << std::endl;
    //     return false;
    // }
    lu_of_A.factorize(A);
    std::cout << lu_of_A.lastErrorMessage() << std::endl;
    // if (lu_of_A.info() != Eigen::ComputationInfo::Success) {
    //     std::cerr<< "factorize failed" << std::endl;
    //     return false;
    // }
    std::cout << "abs det: " << lu_of_A.absDeterminant() << std::endl;

    u = lu_of_A.solve(b);
    std::cout << lu_of_A.lastErrorMessage() << std::endl;
    if (lu_of_A.info() != Eigen::ComputationInfo::Success) {
        std::cerr<< "solving failed" << std::endl;
        return false;
    }
    // if(!lu_of_A.succeeded()) {
    //     std::cerr<< "decomposition failed" << std::endl;
    //     return false;
    // }
    // if(!lu_of_A.solve(b,&u)) {
    //     std::cerr<< "solving failed" << std::endl;
    //     return false;
    // }
    
    return true;
}

bool PoissonBlender::CopyResult(Eigen::Matrix<double, Eigen::Dynamic, 1> &u) {
    const int width = mosaic_.cols;
    const int height = mosaic_.rows;
    const int ch = 3;
    for (int y = 1; y < height - 1; ++y) {
        uchar* pm = mask_.ptr(y) + 1;
        for (int x = 1; x < width - 1; ++x, ++pm) {
            if (*pm != 0) {
                cv::Vec4b & rgba = mosaic_.at<cv::Vec4b>(y, x);
                int idx = mp_[y * (width * ch) + (x * ch)];
                rgba[0] = cv::saturate_cast<uchar>(u[idx + 0] * 255);
                rgba[1] = cv::saturate_cast<uchar>(u[idx + 1] * 255);
                rgba[2] = cv::saturate_cast<uchar>(u[idx + 2] * 255);
            }
        }
    }
    return true;
}

cv::Mat PoissonBlender::GetResult() {
    return mosaic_;
}

}