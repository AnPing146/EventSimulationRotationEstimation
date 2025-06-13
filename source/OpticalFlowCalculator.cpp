// #include <opencv2/video.hpp>

#include "OpticalFlowCalculator.h"
#include <opencv2/video/tracking.hpp>

DenseOpticalFlowCalculator::DenseOpticalFlowCalculator()
    : name_{"DenseOpticalFlowCalculator"} {}

SparseOpticalFlowCalculator::SparseOpticalFlowCalculator()
    : name_{"SparseOpticalFlowCalculator"} {}

FarnebackFlowCalculator::FarnebackFlowCalculator()
    : name_{"FarnebackFlowCalculator"} {}

cv::Mat FarnebackFlowCalculator::calculateFlow(const cv::Mat prev_frame,
                                               const cv::Mat frame)
{
  cv::Mat flow;
  cv::calcOpticalFlowFarneback(prev_frame, frame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  return flow;
}

DISOpticalFlowCalculator::DISOpticalFlowCalculator(
    const int quality)
    : quality_{quality}, name_{"DISOpticalFlowCalculator"}
{
  DISOpticalFlow_ = cv::DISOpticalFlow::create(quality_);

  // dis_param = {sf, ov, it, vio, gamma, alpha}
  // const int dis_prarm[] = {0, 4, 16, 5, 10, 40};
  //DISOpticalFlow_->setPatchSize(4);
  DISOpticalFlow_->setFinestScale(0);
  DISOpticalFlow_->setPatchStride(4);                     // PRESET_MEDIUM:3
  DISOpticalFlow_->setGradientDescentIterations(16);
  DISOpticalFlow_->setVariationalRefinementIterations(5); // PRESET_MEDIUM:5
  DISOpticalFlow_->setVariationalRefinementGamma(10);     // PRESET_MEDIUM:10
  DISOpticalFlow_->setVariationalRefinementAlpha(40);     // PRESET_MEDIUM:10
}

cv::Mat DISOpticalFlowCalculator::calculateFlow(const cv::Mat prev_frame,
                                                const cv::Mat frame)
{
  cv::Mat flow;
  DISOpticalFlow_->calc(prev_frame, frame, flow);

  return flow;
}

LKOpticalFlowCalculator::LKOpticalFlowCalculator(const int max_level,
                                                 const int window_size)
    : max_level_{max_level},
      window_size_{window_size}, name_{"LKOpticalFlowCalculator"} {}

std::vector<cv::Point2f>
LKOpticalFlowCalculator::calculateFlow(cv::Mat prev_frame, cv::Mat frame,
                                       std::vector<cv::Point2f> points,
                                       std::vector<uchar> &status)
{
  if (points.size() == 0)
  {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> new_points;
  std::vector<float> err;

  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

  cv::calcOpticalFlowPyrLK(prev_frame, frame, points, new_points, status, err,
                           cv::Size(window_size_, window_size_), max_level_,
                           criteria);

  return new_points;
}

/*


CudaLKOpticalFlowCalculator::CudaLKOpticalFlowCalculator(const int max_lvl,
                                                         const int window_size)
    : max_lvl_{max_lvl},
      window_size_{window_size},
      name_{"CudaLKOpticalFlowCalculator"} {}

std::vector<cv::Point2f> CudaLKOpticalFlowCalculator::calculateFlow(
    const cv::Mat prev_frame, const cv::Mat frame,
    const std::vector<cv::Point2f> points) {
  if (points.size() == 0) {
    return std::vector<cv::Point2f>();
  }

  cv::cuda::GpuMat frame_gpu(frame);
  cv::cuda::GpuMat prev_frame_gpu(prev_frame);
  cv::cuda::GpuMat points_gpu(points);
  cv::cuda::GpuMat new_points_gpu, status_gpu;

  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ofc =
      cv::cuda::SparsePyrLKOpticalFlow::create(
          cv::Size(window_size_, window_size_), max_lvl_);
  ofc->calc(prev_frame_gpu, frame_gpu, points_gpu, new_points_gpu, status_gpu);

  std::vector<cv::Point2f> new_points(new_points_gpu.cols);
  new_points_gpu.download(new_points);

  return new_points;
}

CudaFarnebackFlowCalculator::CudaFarnebackFlowCalculator()
    : name_{"CudaFarnebackFlowCalculator"} {}

cv::Mat CudaFarnebackFlowCalculator::calculateFlow(const cv::Mat prev_frame,
                                                   const  cv::Mat frame) {
  cv::cuda::GpuMat flow_gpu;

  cv::cuda::GpuMat frame_gpu(frame);
  cv::cuda::GpuMat prev_frame_gpu(prev_frame);

  cv::Ptr<cv::cuda::FarnebackOpticalFlow> ofc =
      cv::cuda::FarnebackOpticalFlow::create(3, 0.5, false, 15);
  ofc->calc(prev_frame_gpu, frame_gpu, flow_gpu);

  cv::Mat flow;
  flow_gpu.download(flow);
  return flow;
}

*/
