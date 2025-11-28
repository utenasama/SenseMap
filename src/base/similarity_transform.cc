//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "base/pose.h"
#include "estimators/similarity_transform.h"
#include "optim/ransac/loransac.h"

#include "base/similarity_transform.h"

namespace sensemap {
namespace {

struct ReconstructionAlignmentEstimator {
    static const int kMinNumSamples = 3;

    typedef const Image* X_t;
    typedef const Image* Y_t;
    typedef Eigen::Matrix3x4d M_t;

    void SetMaxReprojError(const double max_reproj_error) {
        max_squared_reproj_error_ = max_reproj_error * max_reproj_error;
    }

    void SetReconstructions(const Reconstruction* reconstruction1,
                            const Reconstruction* reconstruction2) {
        CHECK_NOTNULL(reconstruction1);
        CHECK_NOTNULL(reconstruction2);
        reconstruction1_ = reconstruction1;
        reconstruction2_ = reconstruction2;
    }

    // Estimate 3D similarity transform from corresponding projection centers.
    std::vector<M_t> Estimate(const std::vector<X_t>& images1,
                              const std::vector<Y_t>& images2) const {
        CHECK_GE(images1.size(), 3);
        CHECK_GE(images2.size(), 3);

        std::vector<Eigen::Vector3d> proj_centers1(images1.size());
        std::vector<Eigen::Vector3d> proj_centers2(images2.size());
        for (size_t i = 0; i < 3; ++i) {
            CHECK_EQ(images1[i]->ImageId(), images2[i]->ImageId());
            proj_centers1[i] = images1[i]->ProjectionCenter();
            proj_centers2[i] = images2[i]->ProjectionCenter();
        }

        SimilarityTransform3 tform12;
        tform12.Estimate(proj_centers1, proj_centers2);

        return {tform12.Matrix().topRows<3>()};
    }

    // For each image, determine the ratio of 3D points that correctly project
    // from one image to the other image and vice versa for the given alignment.
    // The residual is then defined as 1 minus this ratio, i.e., an error
    // threshold of 0.3 means that 70% of the points for that image must reproject
    // within the given maximum reprojection error threshold.
    void Residuals(const std::vector<X_t>& images1,
                   const std::vector<Y_t>& images2,
                   const M_t& alignment12,
                   std::vector<double>* residuals) const {
        CHECK_EQ(images1.size(), images2.size());
        CHECK_NOTNULL(reconstruction1_);
        CHECK_NOTNULL(reconstruction2_);

        const Eigen::Matrix3x4d alignment21 =
            SimilarityTransform3(alignment12).Inverse().Matrix().topRows<3>();

        residuals->resize(images1.size());

        for (size_t i = 0; i < images1.size(); ++i) {
            const auto& image1 = *images1[i];
            const auto& image2 = *images2[i];

            CHECK_EQ(image1.ImageId(), image2.ImageId());
            CHECK_EQ(image1.CameraId(), image2.CameraId());

            const auto& camera1 = reconstruction1_->Camera(image1.CameraId());
            const auto& camera2 = reconstruction2_->Camera(image2.CameraId());

            const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
            const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();

            CHECK_EQ(image1.NumPoints2D(), image2.NumPoints2D());

            size_t num_inliers = 0;
            size_t num_common_points = 0;

            for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
                ++point2D_idx) {
                // Check if both images have a 3D point.

                const auto& point2D1 = image1.Point2D(point2D_idx);
                if (!point2D1.HasMapPoint()) {
                    continue;
                }

                const auto& point2D2 = image2.Point2D(point2D_idx);
                if (!point2D2.HasMapPoint()) {
                    continue;
                }

                num_common_points += 1;

                // Reproject 3D point in image 1 to image 2.
                const Eigen::Vector3d xyz12 =
                    alignment12 *
                    reconstruction1_->MapPoint(point2D1.MapPointId()).XYZ().homogeneous();
                if (CalculateSquaredReprojectionError(point2D2.XY(), xyz12,
                                                      proj_matrix2, camera2) >
                    max_squared_reproj_error_) {
                    continue;
                }

                // Reproject 3D point in image 2 to image 1.
                const Eigen::Vector3d xyz21 =
                    alignment21 *
                    reconstruction2_->MapPoint(point2D2.MapPointId()).XYZ().homogeneous();
                if (CalculateSquaredReprojectionError(point2D1.XY(), xyz21,
                                                      proj_matrix1, camera1) >
                    max_squared_reproj_error_) {
                    continue;
                }

                num_inliers += 1;
            }

            if (num_common_points == 0) {
                (*residuals)[i] = 1.0;
            } else {
                const double negative_inlier_ratio =
                    1.0 - static_cast<double>(num_inliers) /
                              static_cast<double>(num_common_points);
                (*residuals)[i] = negative_inlier_ratio * negative_inlier_ratio;
            }
        }
    }

private:
    double max_squared_reproj_error_ = 0.0;
    const Reconstruction* reconstruction1_ = nullptr;
    const Reconstruction* reconstruction2_ = nullptr;
};



}  // namespace

SimilarityTransform3::SimilarityTransform3() {
    SimilarityTransform3(1, ComposeIdentityQuaternion(),
                        Eigen::Vector3d(0, 0, 0));
}

SimilarityTransform3::SimilarityTransform3(const Eigen::Matrix3x4d& matrix) {
    transform_.matrix().topLeftCorner<3, 4>() = matrix;
}

SimilarityTransform3::SimilarityTransform3(
    const Eigen::Transform<double, 3, Eigen::Affine>& transform)
    : transform_(transform) {}

SimilarityTransform3::SimilarityTransform3(const double scale,
                                           const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec) {
    Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity(4, 4);
    matrix.topLeftCorner<3, 4>() = ComposeProjectionMatrix(qvec, tvec);
    matrix.block<3, 3>(0, 0) *= scale;
    transform_.matrix() = matrix;
}

void SimilarityTransform3::Estimate(const std::vector<Eigen::Vector3d>& src,
                                    const std::vector<Eigen::Vector3d>& dst) {
    transform_.matrix().topLeftCorner<3, 4>() =
        SimilarityTransformEstimator<3>().Estimate(src, dst)[0];
}

SimilarityTransform3 SimilarityTransform3::Inverse() const {
    return SimilarityTransform3(transform_.inverse());
}

void SimilarityTransform3::TransformPoint(Eigen::Vector3d* xyz) const {
    *xyz = transform_ * *xyz;
}

void SimilarityTransform3::TransformPose(Eigen::Vector4d* qvec,
                                         Eigen::Vector3d* tvec) const {
	// Projection matrix P1 projects 3D object points to image plane and thus to
	// 2D image points in the source coordinate system:
	//    x' = P1 * X1
	// 3D object points can be transformed to the destination system by applying
	// the similarity transformation S:
	//    X2 = S * X1
	// To obtain the projection matrix P2 that transforms the object point in 
	// the destination system to the 2D image points, which do not change:
	//    x' = P2 * X2 = P2 * S * X1 = P1 * S^-1 * S * X1 = P1 * I * X1
	// and thus:
	//    P2' = P1 * S^-1
	// Finally, undo the inverse scaling of the rotation matrix:
	//    P2 = s * P2'

	Eigen::Matrix4d src_matrix = Eigen::MatrixXd::Identity(4, 4);
	src_matrix.topLeftCorner<3, 4>() = ComposeProjectionMatrix(*qvec, *tvec);
	Eigen::Matrix4d dst_matrix =
		src_matrix.matrix() * transform_.inverse().matrix();
	dst_matrix *= Scale();

	*qvec = RotationMatrixToQuaternion(dst_matrix.block<3, 3>(0, 0));
	*tvec = dst_matrix.block<3, 1>(0, 3);
}

Eigen::Matrix4d SimilarityTransform3::Matrix() const{
	return transform_.matrix();
}

double SimilarityTransform3::Scale() const{
	return Matrix().block<1, 3>(0, 0).norm();
}

Eigen::Vector4d SimilarityTransform3::Rotation() const{
	return RotationMatrixToQuaternion(Matrix().block<3, 3>(0, 0) / Scale());
}

Eigen::Vector3d SimilarityTransform3::Translation() const{
	return Matrix().block<3, 1>(0, 3);
}


bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& ref_reconstruction,
    const double min_inlier_observations,
    const double max_reproj_error,
    Eigen::Matrix3x4d* alignment) {
    CHECK_GE(min_inlier_observations, 0.0);
    CHECK_LE(min_inlier_observations, 1.0);

    RANSACOptions ransac_options;
    ransac_options.max_error = 1.0 - min_inlier_observations;
    ransac_options.min_inlier_ratio = 0.3;

    LORANSAC<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator>
        ransac(ransac_options);
    ransac.estimator.SetMaxReprojError(max_reproj_error);
    ransac.estimator.SetReconstructions(&src_reconstruction, &ref_reconstruction);
    ransac.local_estimator.SetMaxReprojError(max_reproj_error);
    ransac.local_estimator.SetReconstructions(&src_reconstruction,
                                              &ref_reconstruction);

    const auto& common_image_ids =
        src_reconstruction.FindCommonRegImageIds(ref_reconstruction);

    if (common_image_ids.size() < 3) {
        return false;
    }

    std::vector<const Image*> src_images(common_image_ids.size());
    std::vector<const Image*> ref_images(common_image_ids.size());
    for (size_t i = 0; i < common_image_ids.size(); ++i) {
        src_images[i] = &src_reconstruction.Image(common_image_ids[i]);
        ref_images[i] = &ref_reconstruction.Image(common_image_ids[i]);
    }

    const auto report = ransac.Estimate(src_images, ref_images);

    if (report.success) {
        *alignment = report.model;
    }

    return report.success;
}

bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& ref_reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    const double min_inlier_observations,
    const double max_reproj_error,
    Eigen::Matrix3x4d* alignment) {

    const auto& common_image_ids =
        src_reconstruction.FindCommonRegImageIds(ref_reconstruction);

    std::vector<mappoint_t> common_mappoint_ids1;
    std::vector<mappoint_t> common_mappoint_ids2;
    std::unordered_set<mappoint_t> mappoint_set1;
    std::unordered_set<mappoint_t> mappoint_set2;

    if (common_image_ids.size() > 0) {
        for (const auto& image_id : common_image_ids) {
            const class Image& image1 = src_reconstruction.Image(image_id);
            const class Image& image2 = ref_reconstruction.Image(image_id);

            for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D(); 
                ++point2D_idx) {
                const class Point2D& point2D1 = image1.Point2D(point2D_idx);   
                const class Point2D& point2D2 = image2.Point2D(point2D_idx);
                if (!point2D1.HasMapPoint() || !point2D2.HasMapPoint()) {
                    continue;
                }

                mappoint_t mappoint_id1 = point2D1.MapPointId();
                mappoint_t mappoint_id2 = point2D2.MapPointId();
                if (mappoint_set1.count(mappoint_id1) == 0 &&
                    mappoint_set2.count(mappoint_id2) == 0) {

                    mappoint_set1.insert(mappoint_id1);
                    mappoint_set2.insert(mappoint_id2);

                    common_mappoint_ids1.push_back(mappoint_id1);
                    common_mappoint_ids2.push_back(mappoint_id2);
                }
            }
        }
    } else {
        const EIGEN_STL_UMAP(image_t, Image)& images1 = 
            src_reconstruction.Images();
        const EIGEN_STL_UMAP(image_t, Image)& images2 = 
            ref_reconstruction.Images();
        
        std::vector<image_t> image_ids1;
        for (const auto & image : images1) {
            image_ids1.emplace_back(image.first);
        }
        std::vector<image_t> image_ids2;
        for (const auto & image : images2) {
            image_ids2.emplace_back(image.first);
        }

        // Find common image ids.
        std::vector<std::pair<image_t, image_t> > image_pairs;
        for (const auto & image_id1 : image_ids1) {
            for (const auto & image_id2 : image_ids2) {
                point2D_t num_correspondence =
                    correspondence_graph.NumCorrespondencesBetweenImages(
                        image_id1, image_id2
                    );
                if (num_correspondence) {
                    image_pairs.emplace_back(
                        std::make_pair(image_id1, image_id2)
                    );
                }
            }
        }

        if (image_pairs.size() == 0) {
            return false;
        }
        
        // Find common map points.
        std::vector<const Eigen::Vector3d*> points3D1;
        std::vector<const Eigen::Vector3d*> points3D2;
        for (const auto & image_pair : image_pairs) {
            image_t image_id1 = image_pair.first;
            image_t image_id2 = image_pair.second;
            const class Image& image1 = images1.at(image_id1);
            const class Image& image2 = images2.at(image_id2);
            const FeatureMatches& feature_matches =
                correspondence_graph.FindCorrespondencesBetweenImages(
                    image_id1, image_id2
                );
            for (const auto & feature_match : feature_matches) {
                point2D_t point2D_idx1 = feature_match.point2D_idx1;
                point2D_t point2D_idx2 = feature_match.point2D_idx2;
                const class Point2D& point2D1 = image1.Point2D(point2D_idx1);
                const class Point2D& point2D2 = image2.Point2D(point2D_idx2);
                if (!point2D1.HasMapPoint() || !point2D2.HasMapPoint()) {
                    continue;
                }
                mappoint_t mappoint_id1 = point2D1.MapPointId();
                mappoint_t mappoint_id2 = point2D2.MapPointId();
                const class MapPoint& mappoint1 = 
                    src_reconstruction.MapPoint(mappoint_id1);
                const class MapPoint& mappoint2 = 
                    ref_reconstruction.MapPoint(mappoint_id2);
                if (mappoint1.Error() < max_reproj_error &&
                    mappoint2.Error() < max_reproj_error &&
                    // mappoint1.Track().Length() >= 3 &&
                    // mappoint2.Track().Length() >= 3 &&
                    mappoint_set1.count(mappoint_id1) == 0 &&
                    mappoint_set2.count(mappoint_id2) == 0) {
                
                    mappoint_set1.insert(mappoint_id1);
                    mappoint_set2.insert(mappoint_id2);

                    common_mappoint_ids1.push_back(mappoint_id1);
                    common_mappoint_ids2.push_back(mappoint_id2);

                    points3D1.emplace_back(&mappoint1.XYZ());            
                    points3D2.emplace_back(&mappoint2.XYZ());
                }
            }
        }
    }

    std::cout << "Find " << common_mappoint_ids1.size() 
              << " common mappoints" << std::endl;
    if (common_mappoint_ids1.size() < 3) {
        return false;
    }

    // {
    //     FILE *fp1 = fopen(
    //         StringPrintf("./common_points_%p_%p_1.obj", &src_reconstruction, &ref_reconstruction).c_str(), "w");
    //     FILE *fp2 = fopen(
    //         StringPrintf("./common_points_%p_%p_2.obj", &src_reconstruction, &ref_reconstruction).c_str(), "w");

    //     Eigen::Matrix<double, 3, Eigen::Dynamic> src_mat(3, points3D1.size());
    //     Eigen::Matrix<double, 3, Eigen::Dynamic> dst_mat(3, points3D2.size());
    //     for (size_t i = 0; i < points3D1.size(); ++i) {
    //         src_mat.col(i) = *points3D1[i];
    //         dst_mat.col(i) = *points3D2[i];
    //         fprintf(fp1, "v %f %f %f\n", 
    //             (*points3D1[i])[0], (*points3D1[i])[1], (*points3D1[i])[2]);
    //         fprintf(fp2, "v %f %f %f\n", 
    //             (*points3D2[i])[0], (*points3D2[i])[1], (*points3D2[i])[2]);
    //     }

    //     fclose(fp1);
    //     fclose(fp2);
    // }

    // Optimizing similarity transform.
    RANSACOptions ransac_options;
    ransac_options.max_error = 1.0 - min_inlier_observations;
    ransac_options.min_inlier_ratio = 0.3;

    LORANSAC<PointCloudAlignmentEstimator, PointCloudAlignmentEstimator>
        ransac(ransac_options);
    ransac.estimator.SetMaxReprojError(max_reproj_error);
    ransac.estimator.SetReconstructions(&src_reconstruction, &ref_reconstruction);
    ransac.local_estimator.SetMaxReprojError(max_reproj_error);
    ransac.local_estimator.SetReconstructions(&src_reconstruction,
                                              &ref_reconstruction);

    const auto report = ransac.Estimate(common_mappoint_ids1, common_mappoint_ids2);

    if (report.success) {
        *alignment = report.model;
    }
    if (report.support.num_inliers) { 
        std::cout << "residual error : " << report.support.residual_sum / report.support.num_inliers << std::endl;
    }
    // {
    //     FILE *fp3 = fopen(
    //         StringPrintf("./common_points_%p_%p_1_aligned.obj", &src_reconstruction, &ref_reconstruction).c_str(), "w");
    //     const SimilarityTransform3 tform(*alignment);
    //     for (const auto & point : points3D1) {
    //         Eigen::Vector3d tmpPoint = *point;
    //         tform.TransformPoint(&tmpPoint);

    //         fprintf(fp3, "v %f %f %f\n", tmpPoint[0], tmpPoint[1], tmpPoint[2]);
    //     }
    //     fclose(fp3);
    // }
    return report.success;
}

}  // namespace sensemap
