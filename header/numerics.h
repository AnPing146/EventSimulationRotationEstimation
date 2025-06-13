#ifndef NUMERICS_H
#define NUMERICS_H

//#include <Eigen/Core>

#include <opencv2/core.hpp>
#include "Event.h"

//#include <eigen3/unsupported/Eigen/MatrixFunctions>

cv::Vec3d rotationVectorToYPR(const cv::Vec3d &rot_vec_rad);
cv::Mat gammaCorrect(const cv::Mat& img, float gamma);

/**
 * @brief Rounds a float to 4 decimal places. to keep the precision of the
 *        floating point number.
 */
inline float round4(float x) {
    return std::round(x * 10000.0) / 10000.0;
}
/**
 * @brief Converts a 3D rotation vector to a skew-symmetric matrix (so(3)).
 */
//Eigen::Matrix3f hat(const Eigen::Vector3f &x);
cv::Matx33d hat (const cv::Vec3d &x);


/**
 * @brief Converts a skew-symmetric matrix (so(3)) to a 3D rotation vector.
 */
//Eigen::Vector3f unhat(const Eigen::Matrix3f &x_hat);
cv::Vec3d unhat(const cv::Matx33d &x_hat);

/**
 * @brief Maps a 3D rotation vector to a rotation matrix in SO(3) via the matrix exponential.
 */
//Eigen::Matrix3f SO3(const Eigen::Vector3f &x);
cv::Matx33d SO3(const cv::Vec3d &x);

/**
 * @brief Maps a rotation matrix in SO(3) to a 3D rotation vector via the matrix logarithm.
 */
//Eigen::Vector3f InvSO3(const Eigen::Matrix3f &R);
cv::Vec3d InvSO3(const cv::Matx33d &R);

/**
 * @brief Adds two rotation vectors in SO(3) by composing their corresponding rotation matrices,
 *        then mapping the result back to a rotation vector via logarithmic map.
 * 預期角度合大於pi時，直接使用旋轉向量相加，避免由拉角度轉換為旋轉矩陣後，在-179到179度之間的跳動問題
 *
 * @param x1 First rotation vector
 * @param x2 Second rotation vector
 * @param is_cyclic Whether to bypass group operation when the combined norm exceeds pi
 * @return Resulting rotation vector
 */
//Eigen::Vector3f SO3add(const Eigen::Vector3f &x1, const Eigen::Vector3f &x2, const bool &is_cyclic = false);
cv::Vec3d SO3add(const cv::Vec3d &x1, const cv::Vec3d &x2, const bool &is_cyclic = false);

#endif // NUMERICS_H