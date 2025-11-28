//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "pose.h"
#include "projection.h"
#include "matrix.h"
#include <iostream>
namespace sensemap {

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Vector4d& qvec,
                                          const Eigen::Vector3d& tvec) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = QuaternionToRotationMatrix(qvec);
    proj_matrix.rightCols<1>() = tvec;
    return proj_matrix;
}

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& T) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = R;
    proj_matrix.rightCols<1>() = T;
    return proj_matrix;
}

Eigen::Matrix3x4d InvertProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix) {
    Eigen::Matrix3x4d inv_proj_matrix;
    inv_proj_matrix.leftCols<3>() = proj_matrix.leftCols<3>().transpose();
    inv_proj_matrix.rightCols<1>() = ProjectionCenterFromMatrix(proj_matrix);
    return inv_proj_matrix;
}

Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix) {
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * (svd.matrixV().transpose());
    if (R.determinant() < 0.0) {
        R *= -1.0;
    }
    return R;
}

bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& P, Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R, Eigen::Vector3d* T) {
    Eigen::Matrix3d RR;
    Eigen::Matrix3d QQ;
    DecomposeMatrixRQ(P.leftCols<3>().eval(), &RR, &QQ);

    *R = ComputeClosestRotationMatrix(QQ);

    const double det_K = RR.determinant();
    if (det_K == 0) {
        return false;
    } else if (det_K > 0) {
        *K = RR;
    } else {
        *K = -RR;
    }

    for (int i = 0; i < 3; ++i) {
        if ((*K)(i, i) < 0.0) {
        K->col(i) = -K->col(i);
        R->row(i) = -R->row(i);
        }
    }

    *T = K->triangularView<Eigen::Upper>().solve(P.col(3));
    if (det_K < 0) {
        *T = -(*T);
    }

    return true;
}

Eigen::Vector2d ProjectPointToImage(const Eigen::Vector3d& point3D,
                                    const Eigen::Matrix3x4d& proj_matrix,
                                    const Camera& camera) {
    const Eigen::Vector3d world_point = proj_matrix * point3D.homogeneous();
    return camera.WorldToImage(world_point.hnormalized());
}

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Vector4d& qvec,
                                         const Eigen::Vector3d& tvec,
                                         const Camera& camera) {
    const Eigen::Vector3d proj_point3D =
        QuaternionRotatePoint(qvec, point3D) + tvec;

    
    if(camera.ModelName().compare("SPHERICAL")==0){
        Eigen::Vector3d bearing_point2D = camera.ImageToBearing(point2D);
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;

        if(bearing_point2D(2)>=0){
            local_qvec << camera.Params(3),camera.Params(4),
                          camera.Params(5),camera.Params(6);
            local_tvec << camera.Params(7),camera.Params(8),camera.Params(9);
        }
        else{
            local_qvec << camera.Params(10),camera.Params(11),
                          camera.Params(12),camera.Params(13);
            local_tvec << camera.Params(14),camera.Params(15),camera.Params(16);
        }

        const Eigen::Vector3d proj_point3D_c = 
             QuaternionRotatePoint(local_qvec, proj_point3D) + local_tvec; //////////questionable 

        Eigen::Vector3d bearing_proj_point3D = 
                                        camera.WorldToBearing(proj_point3D_c);
        return (bearing_point2D-bearing_proj_point3D).squaredNorm()*
                camera.FocalLength()*camera.FocalLength();

    }
    else{
        // Check that point is infront of camera.
        if (proj_point3D.z() < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }
        const Eigen::Vector2d proj_point2D =
            camera.WorldToImage(proj_point3D.hnormalized());

        return (proj_point2D - point2D).squaredNorm();
    }
}

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Matrix3x4d& proj_matrix,
                                         const Camera& camera) {
    
    if(camera.ModelName().compare("SPHERICAL")==0){
        Eigen::Vector3d proj_point3D = proj_matrix*(point3D.homogeneous());
              
        Eigen::Vector3d bearing_point2D = camera.ImageToBearing(point2D);

        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;

        if(bearing_point2D(2)>=0){
            local_qvec << camera.Params(3),camera.Params(4),
                          camera.Params(5),camera.Params(6);
            local_tvec << camera.Params(7),camera.Params(8),camera.Params(9);
        }
        else{
            local_qvec << camera.Params(10),camera.Params(11),
                          camera.Params(12),camera.Params(13);
            local_tvec << camera.Params(14),camera.Params(15),camera.Params(16);
        }

        const Eigen::Vector3d proj_point3D_c = 
             QuaternionRotatePoint(local_qvec, proj_point3D) + local_tvec; ////////////??????????questionable

        Eigen::Vector3d bearing_proj_point3D = 
                                        camera.WorldToBearing(proj_point3D_c);

        return (bearing_point2D-bearing_proj_point3D).squaredNorm()*
               camera.FocalLength()*camera.FocalLength();        
    }                                       
    else{
        const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());

        // Check that point is infront of camera.
        if (proj_z < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }

        const double proj_x = proj_matrix.row(0).dot(point3D.homogeneous());
        const double proj_y = proj_matrix.row(1).dot(point3D.homogeneous());
        const double inv_proj_z = 1.0 / proj_z;

        const Eigen::Vector2d proj_point2D = camera.WorldToImage(
            Eigen::Vector2d(inv_proj_z * proj_x, inv_proj_z * proj_y));

        return (proj_point2D - point2D).squaredNorm();
    }
}



double CalculateSquaredReprojectionErrorRig(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Vector4d& qvec,
                                         const Eigen::Vector3d& tvec,
                                         const int local_camera_id,
                                         const Camera& camera) {
    
    const Eigen::Vector3d proj_point3D =
        QuaternionRotatePoint(qvec, point3D) + tvec;
  
   
    if(camera.ModelName().compare("SPHERICAL")==0){
        Eigen::Vector3d bearing_point2D = camera.ImageToBearing(point2D);    
        Eigen::Vector3d bearing_proj_point3D = 
                                        camera.WorldToBearing(proj_point3D);
        return (bearing_point2D-bearing_proj_point3D).squaredNorm()*
                camera.FocalLength()*camera.FocalLength();

    }
    else if(camera.ModelName().compare("UNIFIED") == 0||
            camera.ModelName().compare("OPENCV_FISHEYE") == 0){
        if (proj_point3D.z() < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }
        // const Eigen::Vector2d proj_point2D =
        //     camera.BearingToLocalImage(local_camera_id,proj_point3D.normalized());

        // return (proj_point2D - point2D).squaredNorm();

        Eigen::Vector3d bearing_point2D = camera.LocalImageToBearing(local_camera_id,point2D);  
        double f = 1200; //camera.LocalMeanFocalLength(local_camera_id);    
        Eigen::Vector3d bearing_proj_point3D = proj_point3D.normalized();
        return (bearing_point2D - bearing_proj_point3D).squaredNorm()* f * f;

    }
    else{
        // Check that point is infront of camera.
        if (proj_point3D.z() < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }
        const Eigen::Vector2d proj_point2D =
            camera.WorldToLocalImage(local_camera_id,proj_point3D.hnormalized());

        return (proj_point2D - point2D).squaredNorm();
    }
}


double CalculateSquaredReprojectionErrorRig(const Eigen::Vector2d& point2D,
                                            const Eigen::Vector3d& point3D,
                                            const Eigen::Matrix3x4d& proj_matrix,
                                            const int local_camera_id,
                                            const Camera& camera) {
    
    if(camera.ModelName().compare("SPHERICAL")==0){
        Eigen::Vector3d proj_point3D = proj_matrix*(point3D.homogeneous());
              
        Eigen::Vector3d bearing_point2D = camera.ImageToBearing(point2D);    
        Eigen::Vector3d bearing_proj_point3D = 
                                        camera.WorldToBearing(proj_point3D);

        return (bearing_point2D-bearing_proj_point3D).squaredNorm()*
               camera.FocalLength()*camera.FocalLength();        
    } 
    else if(camera.ModelName().compare("UNIFIED") == 0 || camera.ModelName().compare("OPENCV_FISHEYE") == 0){
    
        const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());
    
        // Check that point is infront of camera.
        if (proj_z < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }

        const double proj_x = proj_matrix.row(0).dot(point3D.homogeneous());
        const double proj_y = proj_matrix.row(1).dot(point3D.homogeneous());
       
        Eigen::Vector3d proj_point3D;
        proj_point3D<<proj_x, proj_y, proj_z;
        

        // const Eigen::Vector2d proj_point2D = camera.BearingToLocalImage(
        //     local_camera_id,
        //     proj_point3D.normalized());

        // return (proj_point2D - point2D).squaredNorm();


        Eigen::Vector3d bearing_point2D = camera.LocalImageToBearing(local_camera_id,point2D);  
        double f = 1200; //camera.LocalMeanFocalLength(local_camera_id);    
        Eigen::Vector3d bearing_proj_point3D = proj_point3D.normalized();
        return (bearing_point2D - bearing_proj_point3D).squaredNorm()* f * f;
    }                              
    else{
        const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());

        // Check that point is infront of camera.
        if (proj_z < std::numeric_limits<double>::epsilon()){
            return std::numeric_limits<double>::max();
        }

        const double proj_x = proj_matrix.row(0).dot(point3D.homogeneous());
        const double proj_y = proj_matrix.row(1).dot(point3D.homogeneous());
        const double inv_proj_z = 1.0 / proj_z;

        const Eigen::Vector2d proj_point2D = camera.WorldToLocalImage(
            local_camera_id,
            Eigen::Vector2d(inv_proj_z * proj_x, inv_proj_z * proj_y));

        return (proj_point2D - point2D).squaredNorm();
    }
}




double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const Camera& camera) {
    
    if(camera.ModelName().compare("SPHERICAL")==0){
        return CalculateAngularErrorSphericalCamera(
                camera.ImageToBearing(point2D),point3D,qvec,tvec,camera);
    }
    else{                             
        return CalculateNormalizedAngularError(camera.ImageToWorld(point2D),
                                               point3D, qvec, tvec);
    }
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const Camera& camera) {
    if(camera.ModelName().compare("SPHERICAL")==0){
        return CalculateAngularErrorSphericalCamera(
                camera.ImageToBearing(point2D),point3D,proj_matrix,camera);
    }                             
    else{
        return CalculateNormalizedAngularError(camera.ImageToWorld(point2D), 
                                            point3D,
                                            proj_matrix);
    }
}



double CalculateAngularErrorRig(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const int local_camera_id,
                             const Camera& camera) {
    
    if(camera.ModelName().compare("SPHERICAL")==0){
        return CalculateAngularErrorSphericalCamera(
                camera.ImageToBearing(point2D),point3D,qvec,tvec);
    }
    else{                             
        return CalculateNormalizedAngularError(
                camera.LocalImageToWorld(local_camera_id,point2D),
                                         point3D, qvec, tvec);
    }
}

double CalculateAngularErrorRig(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const int local_camera_id,
                             const Camera& camera) {
    if(camera.ModelName().compare("SPHERICAL")==0){
        return CalculateAngularErrorSphericalCamera(
                camera.ImageToBearing(point2D),point3D,proj_matrix);
    }                             
    else{
        return CalculateNormalizedAngularError(
                camera.LocalImageToWorld(local_camera_id,point2D), 
                                         point3D,
                                         proj_matrix);
    }
}


double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec) {
    const Eigen::Vector3d ray1 = point2D.homogeneous();
    const Eigen::Vector3d ray2 = QuaternionRotatePoint(qvec, point3D) + tvec;
    return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Matrix3x4d& proj_matrix) {
    const Eigen::Vector3d ray1 = point2D.homogeneous();
    const Eigen::Vector3d ray2 = proj_matrix * point3D.homogeneous();
    return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateAngularErrorSphericalCamera(
                             const Eigen::Vector3d& bearing,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const Camera& camera){
                                  
    const Eigen::Vector3d ray1 = bearing;
    const Eigen::Vector3d proj_point3D = proj_matrix * point3D.homogeneous();

    Eigen::Vector4d local_qvec;
    Eigen::Vector3d local_tvec;

    if (bearing(2) >= 0){
        local_qvec << camera.Params(3), camera.Params(4),
            camera.Params(5), camera.Params(6);
        local_tvec << camera.Params(7), camera.Params(8), camera.Params(9);
    }
    else{
        local_qvec << camera.Params(10), camera.Params(11),
            camera.Params(12), camera.Params(13);
        local_tvec << camera.Params(14), camera.Params(15), camera.Params(16);
    }

    const Eigen::Vector3d ray2 =
        QuaternionRotatePoint(local_qvec, proj_point3D) + local_tvec;

    double cos_theta = ray1.normalized().transpose() * ray2.normalized();
    cos_theta = std::max(-1.0,std::min(1.0,cos_theta));
    return std::acos(cos_theta);                                 
}

double CalculateAngularErrorSphericalCamera(
                             const Eigen::Vector3d& bearing,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const Camera& camera){

    const Eigen::Vector3d ray1 = bearing;
    const Eigen::Vector3d proj_point3D = QuaternionRotatePoint(qvec, point3D) + tvec;

    Eigen::Vector4d local_qvec;
    Eigen::Vector3d local_tvec;

    if (bearing(2) >= 0){
        local_qvec << camera.Params(3), camera.Params(4),
            camera.Params(5), camera.Params(6);
        local_tvec << camera.Params(7), camera.Params(8), camera.Params(9);
    }
    else{
        local_qvec << camera.Params(10), camera.Params(11),
            camera.Params(12), camera.Params(13);
        local_tvec << camera.Params(14), camera.Params(15), camera.Params(16);
    }

    const Eigen::Vector3d ray2 =
        QuaternionRotatePoint(local_qvec, proj_point3D) + local_tvec;

    double cos_theta = ray1.normalized().transpose() * ray2.normalized();
    cos_theta = std::max(-1.0,std::min(1.0,cos_theta));
    return std::acos(cos_theta);                            
}

double CalculateAngularErrorSphericalCamera(
                             const Eigen::Vector3d& bearing,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix){
                                  
    const Eigen::Vector3d ray1 = bearing;
    const Eigen::Vector3d ray2 = proj_matrix * point3D.homogeneous();

    double cos_theta = ray1.normalized().transpose() * ray2.normalized();
    cos_theta = std::max(-1.0,std::min(1.0,cos_theta));
    return std::acos(cos_theta);                                 
}

double CalculateAngularErrorSphericalCamera(
                             const Eigen::Vector3d& bearing,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec){

    const Eigen::Vector3d ray1 = bearing;
    const Eigen::Vector3d ray2 = QuaternionRotatePoint(qvec, point3D) + tvec;

    double cos_theta = ray1.normalized().transpose() * ray2.normalized();
    cos_theta = std::max(-1.0,std::min(1.0,cos_theta));
    return std::acos(cos_theta);                            
}


double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix,
                      const Eigen::Vector3d& point3D) {
    const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());
    return proj_z * proj_matrix.col(2).norm();
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
                           const Eigen::Vector3d& point3D) {
    return proj_matrix.row(2).dot(point3D.homogeneous()) >=
            std::numeric_limits<double>::epsilon();
}




}