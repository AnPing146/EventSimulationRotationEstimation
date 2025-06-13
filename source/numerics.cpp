#include "numerics.h"

//#include <Eigen/Core>
//#include <unsupported/Eigen/MatrixFunctions>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include <math.h>

cv::Matx33d hat(const cv::Vec3d &x)
{
    cv::Matx33d x_hat;
    x_hat << 0, -x(2), x(1),
        x(2), 0, -x(0),
        -x(1), x(0), 0;
    return x_hat;
}

cv::Vec3d unhat(const cv::Matx33d &x_hat)
{
    cv::Vec3d x;
    x << x_hat(2, 1), x_hat(0, 2), x_hat(1, 0);
    return x;
}

cv::Matx33d SO3(const cv::Vec3d &x)
{   
    cv::Matx33d R;
    cv::Rodrigues(x, R);
    return R;
}

cv::Vec3d InvSO3(const cv::Matx33d &R){
    cv::Vec3d x;
    cv::Rodrigues(R, x);
    return x;
}

cv::Vec3d SO3add(const cv::Vec3d &x1, const cv::Vec3d &x2, const bool &is_cyclic){
    if(is_cyclic && cv::norm(x1 + x2) > M_PI){
        std::cout<<"norm > PI\n";
        return x1 + x2;
    }else{
        return InvSO3(SO3(x1) * SO3(x2));
    }
}

cv::Vec3d rotationVectorToYPR(const cv::Vec3d &rot_vec_rad)
{
    // Step 1: 轉成旋轉矩陣
    cv::Mat R;
    cv::Rodrigues(rot_vec_rad, R);

    // Step 2: 根據 R 解出 Yaw (Z), Pitch (Y), Roll (X)
    double yaw, pitch, roll;

    if (std::abs(R.at<double>(2, 0)) < 1.0 - 1e-6)
    {
        pitch = std::asin(-R.at<double>(2, 0));
        roll  = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        yaw   = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        // Gimbal lock 發生（pitch = ±90°）
        pitch = R.at<double>(2, 0) > 0 ? -CV_PI / 2 : CV_PI / 2;
        roll = 0;
        yaw = std::atan2(-R.at<double>(0, 1), R.at<double>(1, 1));
    }

    // Step 3: 轉換成角度
    return cv::Vec3d(
        yaw * 180.0 / CV_PI,
        pitch * 180.0 / CV_PI,
        roll * 180.0 / CV_PI
    ); // 回傳：Yaw, Pitch, Roll (degree)
}

cv::Mat gammaCorrect(const cv::Mat& img, float gamma) {
    CV_Assert(gamma > 0);
    cv::Mat img_f, img_gamma;
    img.convertTo(img_f, CV_32F, 1.0 / 255.0);  // 正規化
    cv::pow(img_f, gamma, img_gamma);
    img_gamma.convertTo(img_gamma, img.type(), 255.0);  // 轉回原格式
    return img_gamma;
}

/*
Eigen::Matrix3f hat(const Eigen::Vector3f &x)
{
    Eigen::Matrix3f x_hat;
    x_hat << 0, -x(2), x(1),
        x(2), 0, -x(0),
        -x(1), x(0), 0;
    return x_hat;
}

Eigen::Vector3f unhat(const Eigen::Matrix3f &x_hat)
{
    Eigen::Vector3f x;
    x << x_hat(2, 1), x_hat(0, 2), x_hat(1, 0);
    return x;
}

Eigen::Matrix3f SO3(const Eigen::Vector3f &x)
{
    return Eigen::Matrix3f(hat(x).exp());
}

Eigen::Vector3f InvSO3(const Eigen::Matrix3f &R)
{
    return Eigen::Vector3f( unhat( R.log() ) );
}

Eigen::Vector3f SO3add(const Eigen::Vector3f &x1, const Eigen::Vector3f &x2, const bool &is_cyclic)
{
    if (is_cyclic && (x1 + x2).norm() > M_PI)
    {
        return x1 + x2;
    }
    else
    {
        return InvSO3(SO3(x1) * SO3(x2));
    }
}
*/