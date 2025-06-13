#ifndef FLOW_COLOR_H
#define FLOW_COLOR_H

#include <opencv2/core.hpp>
#include <string>

/**
 * @brief read flow file in sintel format.
 */
cv::Mat readFlowFile(const std::string &filename);

/**
 * @brief write flow file in sintel format.
 */
void writeFlowFile(const cv::Mat &flow, const std::string &filename);

/**
 * @brief u-v flow to CV_32FC3 mat 
 */
cv::Mat flowToColor(const cv::Mat &flow);

/**
 * @brief draw a flow color map
 */
cv::Mat colorTest();

/**
 * @brief calculate mean end-point error 
 */
double meanEndPointError(const cv::Mat &flow_src, const cv::Mat &flow_gt);

/**
 * @brief 讀取路徑資料夾內的所有 "/frame_*.png" 影像
 * @param filenameExrension 副檔名，例如".png" ".flo"
 */
std::vector<cv::Mat> readFrame(const std::string &path, const std::string &filenameExtension);

#endif