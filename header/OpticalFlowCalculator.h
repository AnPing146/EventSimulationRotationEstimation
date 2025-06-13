#ifndef OPTICAL_FLOW_CALCULATOR_H
#define OPTICAL_FLOW_CALCULATOR_H

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
//#include <opencv2/cudaoptflow.hpp>
#include <string>
#include <vector>

/**
 * @brief Base class for dense optical flow calculators
 */
class DenseOpticalFlowCalculator {
public:
  /**
   * @brief Constructor setting it's name
   */
  DenseOpticalFlowCalculator();

  /**
   * @brief Calculate optical flow
   *
   * @param prev_frame The frame from the previous time step
   * @param frame The fram from the current time step
   */
  virtual cv::Mat calculateFlow(cv::Mat prev_frame, cv::Mat frame) = 0;

  /**
   * @brief Return the name of the opitcal flow calculator
   */
  virtual std::string getName() = 0;

private:
  /// Name of the optical flow calculator
  std::string name_;
};

/**
 * @brief Base class for sparse optical flow calculators
 */
class SparseOpticalFlowCalculator {
public:
  /**
   * @brief Constructor setting the name of the optical flow
   */
  SparseOpticalFlowCalculator();

  virtual std::vector<cv::Point2f>
  /**
   * @brief Calculates the optical flow
   *
   * @param prev_frame Frame from the previous time step
   * @param frame Frame from the current time step
   * @param old_points Points form the prevous calculation
   * @param status 是否找到找到對應舊點的新點
   */
  calculateFlow(cv::Mat prev_frame, cv::Mat frame,
                std::vector<cv::Point2f> old_points,
                std::vector<uchar>& status) = 0;

  /**
   * @brief Returns the name of the optical flow method
   */
  virtual std::string getName() = 0;

private:
  /// Name of the opical flow method
  std::string name_;
};

/**
 * @brief Class implementing the Farneback optical flow calculator.
 */
class FarnebackFlowCalculator : public DenseOpticalFlowCalculator {
public:
  /**
   * @brief Constructor setting the name of the optical flow method.
   */
  FarnebackFlowCalculator();

  /**
   * @brief Calculated the optical flow with the Farneback optical flow
   *        method form OpenCV.
   *
   * @param prev_frame Frame from previous time step
   * @param frame Frame from current time step
   */
  cv::Mat calculateFlow(const cv::Mat prev_frame, const cv::Mat frame) override;

  /**
   * @brief Returns the name of the optical flow method.
   */
  std::string getName() override { return name_; };

private:
  /// Name of the optical flow method
  std::string name_;
};


/**
 * @brief DIS Optical flow calculator
 */
class DISOpticalFlowCalculator : public DenseOpticalFlowCalculator {
public:
  /**
   * @brief Constructor setting the name and the quality level.
   *
   * @param quality Quality level of the DIS optical flow
   */
  DISOpticalFlowCalculator(const int quality);

  /**
   * @brief Calculate the optical flow with:
   *        Fast Optical Flow using Dense Inverse Search:
   *        https://arxiv.org/pdf/1603.03590.pdf
   *
   *        Code taken and adjusted from: https://github.com/tikroeger/OF_DIS
   *
   * @param prev_frame Frame from previous time step
   * @param frame Frame from current time step
   */
  cv::Mat calculateFlow(cv::Mat prev_frame, cv::Mat frame) override;

  /**
   * @brief returns the name of the optical flow method
   */
  std::string getName() override {
    return name_ + "_" + std::to_string(quality_);
  }

private:
  /// DIS optical flow quality
  int quality_;

  /// Name of the optical flow method
  std::string name_;

  ///point to OpenCV DenseOpticalFlow class
  std::shared_ptr<cv::DISOpticalFlow> DISOpticalFlow_;
};

/**
 * @brief Class implementing the Lukas Kanade optical flow calculator.
 */
class LKOpticalFlowCalculator : public SparseOpticalFlowCalculator {
public:
  /**
   * @brief Constructor setting the name, max level and windows size.
   *
   * @param max_level Maximum level
   * @param window_size Window size
   */
  LKOpticalFlowCalculator(const int max_level = 3, const int window_size = 15);

  /**
   * @brief Calculates the flow with the Lukas Canade method from OpenCV.
   *
   * @param prev_frame Frame from the previous time step
   * @param frame Frame from the current time step
   * @param old_points Points from the previous calculation
   * @param status 是否找到找到對應舊點的新點
   */
  std::vector<cv::Point2f>
  calculateFlow(cv::Mat prev_frame, cv::Mat frame,
                std::vector<cv::Point2f> old_points,
                std::vector<uchar>& status) override;

  /**
   * @brief Returns the name of the optical flow method.
   */
  std::string getName() override { return name_; };

private:
  /// Maximum level
  int max_level_;

  /// Window size
  int window_size_;

  /// Name of the optical flow method
  std::string name_;
};

/**
 * @brief Class implementing the Farneback optical flow calculator
 *        with CUDA support.
 */
class CudaFarnebackFlowCalculator : public DenseOpticalFlowCalculator {
 public:
  /**
   * @brief Constructor setting the name.
   */
  CudaFarnebackFlowCalculator();

  /**
   * @brief Calculated the optical flow with the Farneback optical flow
   *        method form OpenCV.
   *
   * @param prev_frame Frame from previous time step
   * @param frame Frame from current time step
   */
  cv::Mat calculateFlow(const cv::Mat prev_frame, const cv::Mat frame) override;

  /**
   * @brief Returns the name of the optical flow method.
   */
  std::string getName() override { return name_; };

 private:
  /// Name of the optical flow method
  std::string name_;
};

/**
 * @brief Class implementing the Lukas Kanade optical flow calculator
 *        with CUDA support.
 */
class CudaLKOpticalFlowCalculator : public SparseOpticalFlowCalculator {
 public:
  /**
   * @brief Constructor initializing the max level and the window size.
   *
   * @param max_lvl Maximal level
   * @param window_size Window size
   */
  CudaLKOpticalFlowCalculator(const int max_lvl = 3,
                              const int window_size = 15);

  /**
   * @brief Calculates the flow with the Lukas Canade method from OpenCV.
   *
   * @param prev_frame Frame from the previous time step
   * @param frame Frame from the current time step
   * @param old_points Points from the previous calculation
   * @param status 是否找到找到對應舊點的新點
   */
  std::vector<cv::Point2f> calculateFlow(
      const cv::Mat prev_frame, const cv::Mat frame,
      const std::vector<cv::Point2f> old_points,
      std::vector<uchar>& status) override;

  /**
   * @brief Returns the name of the optical flow method.
   */
  std::string getName() override { return name_; }

 protected:
  /// Maximum level
  int max_lvl_;

  /// Window size
  int window_size_;

 private:
  /// Name of the optical flow method
  std::string name_;
};

#endif