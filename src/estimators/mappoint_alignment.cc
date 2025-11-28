//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "mappoint_alignment.h"
#include "base/similarity_transform.h"
#include "base/projection.h"

namespace sensemap{

void MapPointAlignmentEstimator::SetReconstruction(
                                 const Reconstruction* reconstruction_src, 
						         const Reconstruction* reconstruction_dst){
    CHECK_NOTNULL(reconstruction_src);												
	CHECK_NOTNULL(reconstruction_dst);
	reconstruction_src_=reconstruction_src;
	reconstruction_dst_=reconstruction_dst;											
}

std::vector<typename MapPointAlignmentEstimator::M_t> 
		MapPointAlignmentEstimator::Estimate(const std::vector<X_t>& src,
						  					 const std::vector<Y_t>& dst){	
	CHECK(src.size()==dst.size());
	CHECK_GE(src.size(), 3);
    CHECK_GE(dst.size(), 3);	

	std::vector<Eigen::Vector3d> points_src(src.size());
	std::vector<Eigen::Vector3d> points_dst(dst.size());

	for(size_t i=0 ; i< src.size(); ++i){
		points_src[i]=reconstruction_src_->MapPoint(src[i]).XYZ();
		points_dst[i]=reconstruction_dst_->MapPoint(dst[i]).XYZ();
	}

	SimilarityTransform3 tform12;
	tform12.Estimate(points_src,points_dst);

	return {tform12.Matrix().topRows<3>()};
}

void MapPointAlignmentEstimator::Residuals(
				 const std::vector<X_t>& points1,
                 const std::vector<Y_t>& points2, const M_t& alignment12,
                 std::vector<double>* residuals){

	CHECK_EQ(points1.size(), points2.size());
    CHECK_NOTNULL(reconstruction_src_);												
	CHECK_NOTNULL(reconstruction_dst_);

	residuals->resize(points1.size());
	
	const Eigen::Matrix3x4d alignment21 =
      	SimilarityTransform3(alignment12).Inverse().Matrix().topRows<3>();

#pragma omp parallel for schedule(dynamic)
	for(size_t i=0; i<points1.size(); ++i){
		const MapPoint& mappoint1= reconstruction_src_->MapPoint(points1[i]);
		const MapPoint& mappoint2= reconstruction_dst_->MapPoint(points2[i]);

	#if 0
		const Eigen::Vector3d xyz12 = alignment12 * mappoint1.XYZ().homogeneous();  
		const Eigen::Vector3d x2 = mappoint2.XYZ();
		(*residuals)[i] = (x2 - xyz12).squaredNorm();
    #else
	    std::vector<TrackElement> track1 = mappoint1.Track().Elements();
		std::vector<TrackElement> track2 = mappoint2.Track().Elements();
		CHECK((!track1.empty())&&(!track2.empty()));
		
		double median_error=0;
		int num_elems=track1.size()+track2.size();
		std::vector<double> reproj_errors1, reproj_errors2;
        //reproject points1 into reconstruction_dst using alignment12  
		const Eigen::Vector3d xyz12 = alignment12*mappoint1.XYZ().homogeneous();  
		for(const auto elem:track2){
			const Image& image2=reconstruction_dst_->Image(elem.image_id);
			const Camera& camera2=reconstruction_dst_->Camera(image2.CameraId());
			const Point2D& point2d2=image2.Point2D(elem.point2D_idx);
			const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();

			if(camera2.NumLocalCameras()>1){
				camera_t local_image_id2 = image2.LocalImageIndices()[elem.point2D_idx];
				double error = CalculateSquaredReprojectionErrorRig(point2d2.XY(),xyz12,proj_matrix2,local_image_id2,camera2);
				reproj_errors1.push_back(error);
			}
			else{
				double error=CalculateSquaredReprojectionError(point2d2.XY(),xyz12,
													proj_matrix2,camera2);
				reproj_errors1.push_back(error);
			}
		}
        //reproject points2 into reconstruction_src using alignment21  
		const Eigen::Vector3d xyz21 = alignment21*mappoint2.XYZ().homogeneous();
		for(const auto elem:track1){
			const Image& image1=reconstruction_src_->Image(elem.image_id);
			const Camera& camera1=reconstruction_src_->Camera(image1.CameraId());
			const Point2D& point2d1=image1.Point2D(elem.point2D_idx);
			const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();

			if(camera1.NumLocalCameras()>1){
				camera_t local_image_id1 = image1.LocalImageIndices()[elem.point2D_idx];
				double error = CalculateSquaredReprojectionErrorRig(point2d1.XY(),xyz21,proj_matrix1,local_image_id1,camera1);
				reproj_errors2.push_back(error);
			}
			else{
				double error=CalculateSquaredReprojectionError(point2d1.XY(),xyz21,
															   proj_matrix1,camera1);
				reproj_errors2.push_back(error);
			}
		}
		//mean_error/=num_elems;
		int nth1 = reproj_errors1.size() / 2;
        std::nth_element(reproj_errors1.begin(), reproj_errors1.begin() + nth1, 
						 reproj_errors1.end());
        int nth2 = reproj_errors2.size() / 2;
        std::nth_element(reproj_errors2.begin(), reproj_errors2.begin() + nth2, 
						 reproj_errors2.end());	
		median_error = std::max(reproj_errors1[nth1],reproj_errors2[nth2]);

		(*residuals)[i]=median_error;
		#endif
	}
}

}//namespace senemap

