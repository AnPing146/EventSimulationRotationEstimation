#include "numerics.cpp"

#include <EventSimulators.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

BasicEventSimulator::BasicEventSimulator() : name_{"BasicRenderer"} {}

std::vector<Event> &BasicEventSimulator::getEvents(const cv::Mat &prev_frame,
                                                   const cv::Mat &frame,
                                                   const float &prev_timestamp,
                                                   const float &timestamp,
                                                   const cv::Rect &roi,
                                                   const bool &getEventFrame,
                                                   cv::Mat &eventFrame,
                                                   const bool &getFlowFrame,
                                                   cv::Mat &flowFrame)
{
  events_.clear();
  return events_;
}

BasicDifferenceEventSimulator::BasicDifferenceEventSimulator(const int c_pos,
                                                             const int c_neg)
    : c_pos_{c_pos}, c_neg_{c_neg}, name_{"BasicDifferenceEventSimulator"} {}

std::vector<Event> &BasicDifferenceEventSimulator::getEvents(const cv::Mat &prev_frame,
                                                             const cv::Mat &frame,
                                                             const float &prev_timestamp,
                                                             const float &timestamp,
                                                             const cv::Rect &roi,
                                                             const bool &getEventFrame,
                                                             cv::Mat &eventFrame,
                                                             const bool &getFlowFrame,
                                                             cv::Mat &flowFrame)
{
  cv::Mat darker, lighter, darker_binary, lighter_binary;
  cv::Mat zeros = cv::Mat::zeros(prev_frame.size(), CV_8UC1);

  events_.clear();

  subtract(prev_frame, frame, darker);
  subtract(frame, prev_frame, lighter);

  cv::threshold(darker, darker_binary, c_neg_, 255, cv::THRESH_BINARY);
  cv::threshold(lighter, lighter_binary, c_pos_, 255, cv::THRESH_BINARY);

  if (roi != cv::Rect(0, 0, 0, 0))
  {
    cv::Mat darker_binary_roi = darker_binary(roi).clone();
    cv::Mat lighter_binary_roi = lighter_binary(roi).clone();
    darker_binary = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
    lighter_binary = cv::Mat::zeros(prev_frame.size(), CV_8UC1);

    darker_binary_roi.copyTo(darker_binary(roi));
    lighter_binary_roi.copyTo(lighter_binary(roi));
  }

  packEvents(lighter_binary, darker_binary, timestamp, events_);

  if (getEventFrame)
  {
    std::vector<cv::Mat> channels;

    channels.push_back(lighter_binary);
    channels.push_back(zeros);
    channels.push_back(darker_binary);

    merge(channels, eventFrame);
  }

  return events_;
}

DifferenceInterpolatedEventSimulator::DifferenceInterpolatedEventSimulator(
    const std::shared_ptr<SparseOpticalFlowCalculator> optical_flow_calculator,
    const int num_inter_frames, const int c_pos, const int c_neg,
    const int c_pos_inter, const int c_neg_inter)
    : num_inter_frames_{num_inter_frames},
      optical_flow_calculator_{optical_flow_calculator},
      name_{"DifferenceInterpolatedEventSimulator"},
      c_pos_{c_pos},
      c_neg_{c_neg},
      c_pos_inter_{c_pos_inter},
      c_neg_inter_{c_neg_inter} {}

void DifferenceInterpolatedEventSimulator::setup(const cv::Size frame_size)
{
  frame_size_ = frame_size;
  std::cout << "Frame size of: " << frame_size << std::endl;
  darker_cached_mask_ = cv::Mat::zeros(frame_size, CV_8UC1);
  lighter_cached_mask_ = cv::Mat::zeros(frame_size, CV_8UC1);
}

std::vector<Event> &DifferenceInterpolatedEventSimulator::getEvents(const cv::Mat &prev_frame,
                                                                    const cv::Mat &frame,
                                                                    const float &prev_timestamp,
                                                                    const float &timestamp,
                                                                    const cv::Rect &roi,
                                                                    const bool &getEventFrame,
                                                                    cv::Mat &eventFrame,
                                                                    const bool &getFlowFrame,
                                                                    cv::Mat &flowFrame)
{
  cv::Size size = prev_frame.size();
  const int x_res = prev_frame.cols;
  const int y_res = prev_frame.rows;
  const int type = prev_frame.type();
  cv::Mat darker(y_res, x_res, type);
  cv::Mat lighter(y_res, x_res, type);
  cv::Mat darker_mask(y_res, x_res, type);
  cv::Mat lighter_mask(y_res, x_res, type);
  cv::Mat lighter_events(y_res, x_res, type);
  cv::Mat darker_events(y_res, x_res, type);
  std::vector<uchar> darker_status, lighter_status;
  cv::Mat tempFrame;
  uchar *lighter_events_ptr = lighter_events.ptr<uchar>();
  uchar *darker_events_ptr = darker_events.ptr<uchar>();
  events_.clear();

  cv::subtract(prev_frame, frame, darker);
  cv::threshold(darker, darker_mask, c_neg_inter_, 255, cv::THRESH_TOZERO);

  cv::subtract(frame, prev_frame, lighter);
  cv::threshold(lighter, lighter_mask, c_pos_inter_, 255, cv::THRESH_TOZERO);

  cv::threshold(darker, darker_events, c_neg_, 255, cv::THRESH_BINARY);
  cv::threshold(lighter, lighter_events, c_pos_, 255, cv::THRESH_BINARY);

  // Option to also add events from the subtract of the current frame and the
  // previous frame
  /*
  std::vector<cv::Point2f> darker_points, lighter_points;
  cv::findNonZero(darker_events, darker_points);
  for (const auto& point : darker_points) {
    events_.emplace_back(point.x, point.y, prev_timestamp, false);
  }
  cv::findNonZero(lighter_events, lighter_points);
  for (const auto& point : lighter_points) {
    events_.emplace_back(point.x, point.y, prev_timestamp, true);
  }
  */

  std::vector<cv::Point2f> prev_darker_points, prev_lighter_points;
  cv::findNonZero(darker_cached_mask_, prev_darker_points);
  cv::findNonZero(lighter_cached_mask_, prev_lighter_points);

  std::vector<cv::Point2f> next_darker_points =
      optical_flow_calculator_->calculateFlow(darker_cached_mask_, darker_mask,
                                              prev_darker_points, darker_status);
  std::vector<cv::Point2f> next_lighter_points =
      optical_flow_calculator_->calculateFlow(
          lighter_cached_mask_, lighter_mask, prev_lighter_points, lighter_status);

  // draw flow frame
  if (getFlowFrame)
  {
    tempFrame = cv::Mat::zeros(size, CV_8UC3);
    for (int i = 0; i < (int)next_darker_points.size(); i++)
    {
      if (!darker_status[i])
        continue;

      cv::arrowedLine(tempFrame, prev_darker_points[i], next_darker_points[i], cv::Scalar(0, 255, 0));
    }

    for (int i = 0; i < (int)next_lighter_points.size(); i++)
    {
      if (!lighter_status[i])
        continue;

      cv::arrowedLine(tempFrame, prev_lighter_points[i], next_lighter_points[i], cv::Scalar(0, 255, 0));
    }
    flowFrame = tempFrame;
  }

  // calculate the interpolated points
  for (int j = 0; j < num_inter_frames_; j++)
  {
    float alpha =
        static_cast<float>(j + 1) / static_cast<float>(num_inter_frames_ + 1);
    float current_timestamp =
        prev_timestamp + round4((timestamp - prev_timestamp) * alpha);

    // calcuate the lighter points
    for (std::size_t i = 0; i < next_lighter_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_lighter_points[i] +
          (next_lighter_points[i] - prev_lighter_points[i]) * alpha;
      // check if in bounds with ROI
      if (roi == cv::Rect(0, 0, 0, 0))
      {
        if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
            inter_point.x < 0 || inter_point.y < 0)
        {
          continue;
        }
        if (getEventFrame)
        {
          *(lighter_events_ptr + cvRound(inter_point.y) * x_res + cvRound(inter_point.x)) = 255;
          // lighter_events.at<uchar>(cvRound(inter_point.y), cvRound(inter_point.x)) = 255;
        }
        events_.emplace_back(inter_point.x, inter_point.y, current_timestamp, true);
      }
      else
      {
        if (inter_point.x > (roi.x + roi.width - 1) || inter_point.y > (roi.y + roi.height - 1) ||
            inter_point.x < roi.x || inter_point.y < roi.y)
        {
          continue;
        }
        if (getEventFrame)
        {
          *(lighter_events_ptr + cvRound(inter_point.y) * x_res + cvRound(inter_point.x)) = 255;
          // lighter_events.at<uchar>(cvRound(inter_point.y), cvRound(inter_point.x)) = 255;
        }
        events_.emplace_back(inter_point.x, inter_point.y, current_timestamp,
                             true);
      }
    }

    // calculate the darker points
    for (std::size_t i = 0; i < next_darker_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_darker_points[i] +
          (next_darker_points[i] - prev_darker_points[i]) * alpha;
      // check if in bounds with ROI
      if (roi == cv::Rect(0, 0, 0, 0))
      {
        if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
            inter_point.x < 0 || inter_point.y < 0)
        {
          continue;
        }
        if (getEventFrame)
        {
          *(darker_events_ptr + cvRound(inter_point.y) * x_res + cvRound(inter_point.x)) = 255;
        }
        events_.emplace_back(inter_point.x, inter_point.y, current_timestamp, false);
      }
      else
      {
        if (inter_point.x > (roi.x + roi.width - 1) || inter_point.y > (roi.y + roi.height - 1) ||
            inter_point.x < roi.x || inter_point.y < roi.y)
        {
          continue;
        }
        if (getEventFrame)
        {
          *(darker_events_ptr + cvRound(inter_point.y) * x_res + cvRound(inter_point.x)) = 255;
        }
        events_.emplace_back(inter_point.x, inter_point.y, current_timestamp, false);
      }
    }
  }

  darker_cached_mask_ = darker_mask;
  lighter_cached_mask_ = lighter_mask;

  if (getEventFrame)
  {
    cv::Mat zeros = cv::Mat::zeros(size, CV_8UC1);
    std::vector<cv::Mat> channels;

    channels.push_back(lighter_events);
    channels.push_back(zeros);
    channels.push_back(darker_events);
    /*
        if (roi == cv::Rect(0, 0, 0, 0))
        {
          channels.push_back(lighter_events);
          channels.push_back(zeros);
          channels.push_back(darker_events);
        }
        else
        {
          channels.push_back(cv::Mat(lighter_events, roi));
          channels.push_back(cv::Mat(zeros, roi));
          channels.push_back(cv::Mat(darker_events, roi));
        }
    */
    merge(channels, eventFrame);
  }

  return events_;
}

DenseInterpolatedEventSimulator::DenseInterpolatedEventSimulator(
    const std::shared_ptr<DenseOpticalFlowCalculator> optical_flow_calculator,
    const int num_inter_frames, const int c_pos, const int c_neg)
    : num_inter_frames_{num_inter_frames},
      optical_flow_calculator_{optical_flow_calculator},
      name_{"DenseInterpolatedEventSimulator"}, c_pos_{c_pos}, c_neg_{c_neg} {}

std::vector<Event> &DenseInterpolatedEventSimulator::getEvents(const cv::Mat &prev_frame,
                                                               const cv::Mat &frame,
                                                               const float &prev_timestamp,
                                                               const float &timestamp,
                                                               const cv::Rect &roi,
                                                               const bool &getEventFrame,
                                                               cv::Mat &eventFrame,
                                                               const bool &getFlowFrame,
                                                               cv::Mat &flowFrame)
{
  cv::Mat prev_inter_frame = prev_frame;
  cv::Mat darker_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  cv::Mat lighter_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  cv::Mat flow = optical_flow_calculator_->calculateFlow(prev_frame, frame);
  cv::Point2f *interflow_ptr, *map_ptr, flow_temp;
  events_.clear();

  // draw flow frame
  if (getFlowFrame)
  {
    cv::Mat flow_uv[2];
    cv::Mat hsv_split[3], hsv;
    cv::Mat mag, ang;

    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = cv::Mat::ones(ang.size(), ang.type());
    hsv_split[2] = mag;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, flowFrame, cv::COLOR_HSV2BGR);
  }

  // calculate the interpolated event points
  for (int j = 1; j < num_inter_frames_ + 2; j++) ////第0次產生zero flow,可以刪掉這次無用的計算
  {
    float alpha =
        static_cast<float>(j) / static_cast<float>(num_inter_frames_ + 1.0);
    // std::cout<<"["<<j<<"]alpha: "<<alpha<<"\n";
    float current_timestamp =
        prev_timestamp + round4((timestamp - prev_timestamp) * alpha);
    // std::cout<<"["<<j<<"]current_timestamp: "<<current_timestamp<<"\n";
    cv::Mat interflow = alpha * flow;
    cv::Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
      interflow_ptr = interflow.ptr<cv::Point2f>(y);
      map_ptr = map.ptr<cv::Point2f>(y);
      for (int x = 0; x < map.cols; ++x)
      {
        flow_temp = *(interflow_ptr + x);
        *(map_ptr + x) = cv::Point2f(x - flow_temp.x, y - flow_temp.y); // remap()是逆映射，所以要反向插值
        // cv::Point2f f = interflow.at<cv::Point2f>(y, x);
        // map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
      }
    }

    cv::Mat inter_frame;
    // cv::Mat flow_parts[2];
    // cv::split(map, flow_parts);
    cv::remap(prev_frame, inter_frame, map, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REPLICATE); 

    cv::Mat darker, lighter, darker_binary, lighter_binary;

    cv::subtract(prev_inter_frame, inter_frame, darker);
    cv::subtract(inter_frame, prev_inter_frame, lighter);

    // distinguish light or dark event
    cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
    cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);

    //cv::namedWindow("interFrame", cv::WINDOW_NORMAL);
    //cv::imshow("interFrame", inter_frame);
    //cv::Mat temp = prev_frame + lighter - darker;
    //cv::namedWindow("prevFrame plus lighter minus darker", cv::WINDOW_NORMAL);
    //cv::imshow("prevFrame plus lighter minus darker", temp);
    //cv::namedWindow("lighter", cv::WINDOW_NORMAL);
    //cv::namedWindow("darker", cv::WINDOW_NORMAL);
    //cv::imshow("darker", darker);
    //cv::imshow("lighter", lighter);
    //cv::waitKey(0);

    if (roi != cv::Rect(0, 0, 0, 0))
    {
      cv::Mat lighter_roi = lighter(roi).clone();
      cv::Mat darker_roi = darker(roi).clone();

      lighter = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
      darker = cv::Mat::zeros(prev_frame.size(), CV_8UC1);

      lighter_roi.copyTo(lighter(roi));
      darker_roi.copyTo(darker(roi));
    }

    if (getEventFrame)
    {
      cv::add(darker, darker_frame, darker_frame);          // 迭代多次插值到同一張影像
      cv::add(lighter, lighter_frame, lighter_frame);
    }
    packEvents(lighter, darker, current_timestamp, events_);

    prev_inter_frame = inter_frame;

    /*
        // distinguish light or dark event
        cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
        cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);

        if (getEventFrame)
        {
          cv::add(darker, darker_frame, darker_frame); // 迭代多次插值到同一張影像
          cv::add(lighter, lighter_frame, lighter_frame);
        }
        packEvents(lighter, darker, current_timestamp, roi, events_);

        prev_inter_frame = inter_frame;
    */
  }

  if (getEventFrame)
  {
    cv::Mat zeros = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
    std::vector<cv::Mat> channels;

    channels.push_back(lighter_frame); //////lighter_frame
    channels.push_back(zeros);
    channels.push_back(darker_frame); //////darker_frame

    merge(channels, eventFrame);
  }

  return events_;
}

SparseInterpolatedEventSimulator::SparseInterpolatedEventSimulator(
    const std::shared_ptr<SparseOpticalFlowCalculator> optical_flow_calculator,
    const int num_inter_frames, const int c_pos, const int c_neg)
    : num_inter_frames_{num_inter_frames},
      optical_flow_calculator_{optical_flow_calculator},
      name_{"SparseInterpolatedEventSimulator"},
      c_pos_{c_pos},
      c_neg_{c_neg} {}

std::vector<Event> &SparseInterpolatedEventSimulator::getEvents(const cv::Mat &prev_frame,
                                                                const cv::Mat &frame,
                                                                const float &prev_timestamp,
                                                                const float &timestamp,
                                                                const cv::Rect &roi,
                                                                const bool &getEventFrame,
                                                                cv::Mat &eventFrame,
                                                                const bool &getFlowFrame,
                                                                cv::Mat &flowFrame)
{
  cv::Size size = prev_frame.size();
  int x_res = prev_frame.cols;
  int y_res = prev_frame.rows;
  cv::Mat darker_frame = cv::Mat::zeros(size, CV_8UC1);
  cv::Mat lighter_frame = cv::Mat::zeros(size, CV_8UC1);
  cv::Mat mask;
  cv::Mat tempFrame;
  std::vector<uchar> status;
  events_.clear();

  cv::absdiff(prev_frame, frame, mask);
  cv::threshold(mask, mask, c_pos_, 255, cv::THRESH_BINARY);

  std::vector<cv::Point2f> prev_points;
  cv::findNonZero(mask, prev_points);

  std::vector<cv::Point2f> next_points =
      optical_flow_calculator_->calculateFlow(prev_frame, frame, prev_points, status);
  cv::Mat prev_inter_frame = prev_frame;

  // draw flow frame
  if (getFlowFrame)
  {
    tempFrame = cv::Mat::zeros(prev_frame.size(), CV_8UC3);
    for (int i = 0; i < (int)next_points.size(); i += 5)
    {
      if (!status[i])
        continue;
      cv::arrowedLine(tempFrame, prev_points[i], next_points[i], cv::Scalar(0, 255, 0));
      // cv::circle(tempFrame, next_points[i], 2, cv::Scalar(0,255,0), -1);
      // cv::line(tempFrame, prev_points[i], next_points[i], cv::Scalar(0,255,0));
    }
    flowFrame = tempFrame;
    // std::cout<<"tempFrame.size(): "<<tempFrame.size()<<"\n";
  }

  // calculate the interpolated points
  for (int j = 0; j < num_inter_frames_ + 2; j++)
  {
    float alpha =
        static_cast<float>(j) / static_cast<float>(num_inter_frames_ + 1);
    float current_timestamp =
        prev_timestamp + round4((timestamp - prev_timestamp) * alpha);
    cv::Mat inter_frame;
    prev_frame.copyTo(inter_frame);

    for (uint i = 0; i < next_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_points[i] + (next_points[i] - prev_points[i]) * alpha;

      // check if in bounds
      // if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
      //    inter_point.x < 0 || inter_point.y < 0)
      //{
      //  continue;
      //}
      int x = static_cast<int>(inter_point.x);
      int y = static_cast<int>(inter_point.y);
      int x0 = static_cast<int>(prev_points[i].x);
      int y0 = static_cast<int>(prev_points[i].y);

      if (x >= x_res || y >= y_res || x < 0 || y < 0)
      {
        continue;
      }

      *(inter_frame.ptr<uchar>(y) + x) = *(prev_frame.ptr<uchar>(y0) + x0);
      // inter_frame.at<uchar>(inter_point) = prev_frame.at<uchar>(prev_points[i]);
      //  time.at<uchar>(inter_point) = (int) (alpha * 127.0);
    }

    cv::Mat darker, lighter, darker_binary, lighter_binary;

    cv::subtract(prev_inter_frame, inter_frame, darker);
    cv::subtract(inter_frame, prev_inter_frame, lighter);

    // distinguish light or dark event
    cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
    cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);

    if (roi != cv::Rect(0, 0, 0, 0))
    {
      cv::Mat lighter_roi = lighter(roi).clone();
      cv::Mat darker_roi = darker(roi).clone();
      lighter = cv::Mat::zeros(lighter.size(), CV_8UC1);
      darker = cv::Mat::zeros(darker.size(), CV_8UC1);

      lighter_roi.copyTo(lighter(roi));
      darker_roi.copyTo(darker(roi));
    }

    if (getEventFrame)
    {
      cv::add(darker, darker_frame, darker_frame); // 迭代多次插值到同一張影像
      cv::add(lighter, lighter_frame, lighter_frame);
    }

    packEvents(lighter, darker, current_timestamp, events_);

    prev_inter_frame = inter_frame;
  }

  if (getEventFrame)
  {
    cv::Mat zeros = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
    std::vector<cv::Mat> channels;

    channels.push_back(lighter_frame);
    channels.push_back(zeros);
    channels.push_back(darker_frame);

    merge(channels, eventFrame);
  }

  return events_;
}

/*
 * std::vector<Event>& BasicEventSimulator::getEvents(const cv::Mat prev_frame,
                                                   const cv::Mat frame,
                                                   const unsigned int prev_timestamp,
                                                   const unsigned int timestamp,
                                                   int& num_frames) {
  events_.clear();
  return events_;
}
std::vector<cv::Mat>& BasicEventSimulator::getEventFrame(const cv::Mat prev_frame,
                                                         const cv::Mat frame,
                                                         int& num_frames,
                                                         const bool getFlowFrame,
                                                         cv::Mat& flowFrame) {
  num_frames = 1;
  out_frames_.resize(1);
  out_frames_.at(0) = prev_frame;
  return out_frames_;
}

std::vector<Event>& BasicDifferenceEventSimulator::getEvents(
    const cv::Mat prev_frame, const cv::Mat frame, const unsigned int prev_timestamp,
    const unsigned int timestamp, int& num_frames) {
  num_frames = 1;

  cv::Mat darker, lighter, darker_binary, lighter_binary;

  subtract(prev_frame, frame, darker);
  cv::threshold(darker, darker_binary, c_pos_, 255, cv::THRESH_BINARY);

  subtract(frame, prev_frame, lighter);
  cv::threshold(lighter, lighter_binary, c_neg_, 255, cv::THRESH_BINARY);

  events_.clear();
  packEvents(lighter_binary, darker_binary, timestamp, events_);

  return events_;
}

std::vector<cv::Mat>&
BasicDifferenceEventSimulator::getEventFrame(const cv::Mat prev_frame, const cv::Mat frame,
                                             int& num_frames,
                                             const bool getFlowFrame,
                                             cv::Mat& flowFrame) {
  num_frames = 1;

  cv::Mat darker, lighter, darker_binary, lighter_binary;

  out_frames_.resize(num_frames);

  subtract(prev_frame, frame, darker);
  cv::threshold(darker, darker_binary, c_pos_, 255, cv::THRESH_BINARY);

  subtract(frame, prev_frame, lighter);
  cv::threshold(lighter, lighter_binary, c_neg_, 255, cv::THRESH_BINARY);

  cv::Mat zeros = cv::Mat::zeros(darker_binary.size(), CV_8UC1);

  std::vector<cv::Mat> channels;
  channels.push_back(lighter_binary);
  channels.push_back(zeros);
  channels.push_back(darker_binary);
  merge(channels, out_frames_.at(0));

  return out_frames_;
}

std::vector<Event> &DifferenceInterpolatedEventSimulator::getEvents(
    const cv::Mat prev_frame, const cv::Mat frame,
    const unsigned int prev_timestamp, const unsigned int timestamp,
    int &num_frames)
{
  num_frames = 1;
  const auto &size = prev_frame.size;
  const auto &cols = prev_frame.cols;
  const auto &rows = prev_frame.rows;
  const int type = prev_frame.type();
  cv::Mat darker(rows, cols, type);
  cv::Mat lighter(rows, cols, type);
  cv::Mat darker_mask(rows, cols, type);
  cv::Mat lighter_mask(rows, cols, type);
  cv::Mat lighter_events(rows, cols, type);
  cv::Mat darker_events(rows, cols, type);
  std::vector<uchar> darker_status, lighter_status;

  events_.clear();

  cv::subtract(prev_frame, frame, darker);
  cv::threshold(darker, darker_mask, c_pos_inter_, 255, cv::THRESH_TOZERO);

  cv::subtract(frame, prev_frame, lighter);
  cv::threshold(lighter, lighter_mask, c_neg_inter_, 255, cv::THRESH_TOZERO);

  cv::threshold(darker, darker_events, c_pos_, 255, cv::THRESH_BINARY);
  cv::threshold(lighter, lighter_events, c_neg_, 255, cv::THRESH_BINARY);

  // Option to also add events from the subtract of the current frame and the
  // previous frame

  //std::vector<cv::Point2f> darker_points, lighter_points;
  //cv::findNonZero(darker_events, darker_points);
  //for (const auto& point : darker_points) {
  //  events_.emplace_back(point.x, point.y, prev_timestamp, false);
  //}
  //cv::findNonZero(lighter_events, lighter_points);
  //for (const auto& point : lighter_points) {
  //  events_.emplace_back(point.x, point.y, prev_timestamp, true);
  //}


  std::vector<cv::Point2f> prev_darker_points, prev_lighter_points;
  cv::findNonZero(darker_cached_mask_, prev_darker_points);
  cv::findNonZero(lighter_cached_mask_, prev_lighter_points);

  std::vector<cv::Point2f> next_darker_points =
      optical_flow_calculator_->calculateFlow(darker_cached_mask_, darker_mask,
                                              prev_darker_points, darker_status);
  std::vector<cv::Point2f> next_lighter_points =
      optical_flow_calculator_->calculateFlow(
          lighter_cached_mask_, lighter_mask, prev_lighter_points, lighter_status);

  int x_res = darker_mask.cols;
  int y_res = darker_mask.rows;

  for (int j = 0; j < num_inter_frames_; j++)
  {
    float alpha =
        static_cast<double>(j + 1) / static_cast<double>(num_inter_frames_ + 1);

    for (std::size_t i = 0; i < next_lighter_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_lighter_points[i] +
          (next_lighter_points[i] - prev_lighter_points[i]) * alpha;
      unsigned int current_timestamp =
          prev_timestamp + (timestamp - prev_timestamp) * alpha;
      // check if in bounds
      if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
          inter_point.x < 0 || inter_point.y < 0)
      {
        continue;
      }
      events_.emplace_back(inter_point.x, inter_point.y, current_timestamp,
                           true);
    }

    for (std::size_t i = 0; i < next_darker_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_darker_points[i] +
          (next_darker_points[i] - prev_darker_points[i]) * alpha;
      unsigned int current_timestamp =
          prev_timestamp + (timestamp - prev_timestamp) * alpha;
      // check if in bounds
      if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
          inter_point.x < 0 || inter_point.y < 0)
      {
        continue;
      }
      events_.emplace_back(inter_point.x, inter_point.y, current_timestamp,
                           false);
    }
  }

  darker_cached_mask_ = darker_mask;
  lighter_cached_mask_ = lighter_mask;

  return events_;
}

std::vector<cv::Mat> &DifferenceInterpolatedEventSimulator::getEventFrame(
    const cv::Mat prev_frame, const cv::Mat frame, int &num_frames,
    const bool getFlowFrame,
    cv::Mat &flowFrame)
{
  num_frames = 1;
  const auto &size = prev_frame.size;
  const auto &cols = prev_frame.cols;
  const auto &rows = prev_frame.rows;
  const int type = prev_frame.type();
  cv::Mat darker(rows, cols, type);
  cv::Mat lighter(rows, cols, type);
  cv::Mat darker_mask(rows, cols, type);
  cv::Mat lighter_mask(rows, cols, type);
  cv::Mat lighter_events(rows, cols, type);
  cv::Mat darker_events(rows, cols, type);
  std::vector<uchar> darker_status, lighter_status;
  cv::Mat tempFrame;

  out_frames_.resize(num_frames);

  cv::subtract(prev_frame, frame, darker);
  cv::threshold(darker, darker_mask, c_pos_inter_, 255, cv::THRESH_TOZERO);

  cv::subtract(frame, prev_frame, lighter);
  cv::threshold(lighter, lighter_mask, c_neg_inter_, 255, cv::THRESH_TOZERO);

  // Option to also add events from the subtract of the current frame and the
  // previous frame
  // cv::threshold(darker, darker_events, c_pos_, 255, cv::THRESH_BINARY);
  // cv::threshold(lighter, lighter_events, c_neg_, 255, cv::THRESH_BINARY);

  std::vector<cv::Point2f> prev_darker_points, prev_lighter_points;
  cv::findNonZero(darker_cached_mask_, prev_darker_points);
  cv::findNonZero(lighter_cached_mask_, prev_lighter_points);

  std::vector<cv::Point2f> next_darker_points =
      optical_flow_calculator_->calculateFlow(darker_cached_mask_, darker_mask,
                                              prev_darker_points, darker_status);
  std::vector<cv::Point2f> next_lighter_points =
      optical_flow_calculator_->calculateFlow(
          lighter_cached_mask_, lighter_mask, prev_lighter_points, lighter_status);

  // draw flow frame
  if (getFlowFrame)
  {
    tempFrame = cv::Mat::zeros(prev_frame.size(), CV_8UC3);
    for (int i = 0; i < (int)next_darker_points.size(); i++)
    {
      if (!darker_status[i])
        continue;

      cv::arrowedLine(tempFrame, prev_darker_points[i], next_darker_points[i], cv::Scalar(0, 255, 0));
    }

    for (int i = 0; i < (int)next_lighter_points.size(); i++)
    {
      if (!lighter_status[i])
        continue;

      cv::arrowedLine(tempFrame, prev_lighter_points[i], next_lighter_points[i], cv::Scalar(0, 255, 0));
    }
    flowFrame = tempFrame;
  }
  int x_res = darker_mask.cols;
  int y_res = darker_mask.rows;
  cv::Mat zeros = cv::Mat::zeros(lighter_mask.size(), CV_8UC1);

  for (int j = 0; j < num_inter_frames_; j++)
  {
    float alpha =
        static_cast<double>(j + 1) / static_cast<double>(num_inter_frames_ + 1);

    for (uint i = 0; i < next_lighter_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_lighter_points[i] +
          (next_lighter_points[i] - prev_lighter_points[i]) * alpha;
      // check if in bounds
      if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
          inter_point.x < 0 || inter_point.y < 0)
        continue;
      lighter_events.at<uchar>(inter_point) = 255;
    }

    for (uint i = 0; i < next_darker_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_darker_points[i] +
          (next_darker_points[i] - prev_darker_points[i]) * alpha;
      // check if in bounds
      if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
          inter_point.x < 0 || inter_point.y < 0)
        continue;
      darker_events.at<uchar>(inter_point) = 255;
    }
  }
  darker_cached_mask_ = darker_mask;
  lighter_cached_mask_ = lighter_mask;

  std::vector<cv::Mat> channels;
  channels.push_back(lighter_events);
  channels.push_back(zeros);
  channels.push_back(darker_events);
  cv::merge(channels, out_frames_.at(0));

  return out_frames_;
}

std::vector<Event> &DenseInterpolatedEventSimulator::getEvents(
    cv::Mat prev_frame, cv::Mat frame, unsigned int prev_timestamp,
    unsigned int timestamp, int &num_frames)
{
  num_frames = 1;

  cv::Mat flow = optical_flow_calculator_->calculateFlow(prev_frame, frame);
  cv::Mat prev_inter_frame = prev_frame;

  events_.clear();
  //std::vector<cv::Mat> darker_frames;  //沒用到
  //std::vector<cv::Mat> lighter_frames;
  for (int j = 0; j < num_inter_frames_ + 2; j++)
  {
    float alpha =
        static_cast<double>(j) / static_cast<double>(num_inter_frames_ + 1.0);
    unsigned int current_timestamp =
        prev_timestamp + alpha * (timestamp - prev_timestamp);
    cv::Mat interflow = alpha * flow;
    cv::Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
      for (int x = 0; x < map.cols; ++x)
      {
        cv::Point2f f = interflow.at<cv::Point2f>(y, x);
        map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
      }
    }

    cv::Mat inter_frame;
    cv::Mat flow_parts[2];
    cv::split(map, flow_parts);
    cv::remap(prev_frame, inter_frame, flow_parts[0], flow_parts[1],
              cv::INTER_LINEAR);

    cv::Mat darker, lighter, darker_binary, lighter_binary;

    cv::subtract(prev_inter_frame, inter_frame, darker);
    cv::subtract(inter_frame, prev_inter_frame, lighter);

    cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
    cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);

    packEvents(lighter, darker, current_timestamp, events_);

    prev_inter_frame = inter_frame;
  }

  return events_;
}

std::vector<cv::Mat> &
DenseInterpolatedEventSimulator::getEventFrame(const cv::Mat prev_frame,
                                               const cv::Mat frame, int &num_frames,
                                               const bool getFlowFrame,
                                               cv::Mat &flowFrame)
{
  num_frames = 1;

  cv::Mat flow = optical_flow_calculator_->calculateFlow(prev_frame, frame);
  cv::Mat prev_inter_frame = prev_frame;

  // draw flow frame
  if (getFlowFrame)
  {
    cv::Mat flow_uv[2];
    cv::Mat hsv_split[3], hsv;
    cv::Mat mag, ang;

    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = cv::Mat::ones(ang.size(), ang.type());
    hsv_split[2] = mag;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, flowFrame, cv::COLOR_HSV2BGR);
  }

  cv::Mat darker_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  cv::Mat lighter_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);

  // std::vector<cv::Mat> darker_frames;
  // std::vector<cv::Mat> lighter_frames;
  for (int j = 0; j < num_inter_frames_ + 2; j++)
  {
    float alpha =
        static_cast<double>(j) / static_cast<double>(num_inter_frames_ + 1.0);
    cv::Mat interflow = alpha * flow;
    cv::Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
      for (int x = 0; x < map.cols; ++x)
      {
        cv::Point2f f = interflow.at<cv::Point2f>(y, x);
        map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
      }
    }

    cv::Mat inter_frame;
    cv::Mat flow_parts[2];
    cv::split(map, flow_parts);
    cv::remap(prev_frame, inter_frame, flow_parts[0], flow_parts[1],
              cv::INTER_LINEAR);

    cv::Mat darker, lighter, darker_binary, lighter_binary;

    cv::subtract(prev_inter_frame, inter_frame, darker);
    cv::subtract(inter_frame, prev_inter_frame, lighter);

    cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
    cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);

    cv::add(darker, darker_frame, darker_frame); // 迭代多次插值到同一張影像
    cv::add(lighter, lighter_frame, lighter_frame);

    prev_inter_frame = inter_frame;
  }

  out_frames_.resize(num_frames);

  std::vector<cv::Mat> channels;
  cv::Mat zeros = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  channels.push_back(lighter_frame);
  channels.push_back(zeros);
  channels.push_back(darker_frame);
  cv::merge(channels, out_frames_.at(0));

  return out_frames_;
}

std::vector<Event> &SparseInterpolatedEventSimulator::getEvents(
  const cv::Mat prev_frame, const cv::Mat frame,
  const unsigned int prev_timestamp, const unsigned int timestamp,
  int &num_frames)
{
num_frames = 1;
cv::Mat mask;
std::vector<uchar> status;

cv::absdiff(prev_frame, frame, mask);
cv::threshold(mask, mask, c_pos_, 255, cv::THRESH_BINARY);

std::vector<cv::Point2f> prev_points;
cv::findNonZero(mask, prev_points);

std::vector<cv::Point2f> next_points =
    optical_flow_calculator_->calculateFlow(prev_frame, frame, prev_points, status);

cv::Mat prev_inter_frame = prev_frame;

events_.clear();
cv::Mat darker_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
cv::Mat lighter_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
int x_res = mask.cols;
int y_res = mask.rows;

for (int j = 0; j < num_inter_frames_ + 2; j++)
{
  float alpha =
      static_cast<double>(j) / static_cast<double>(num_inter_frames_ + 1);
  unsigned int current_timestamp =
      prev_timestamp + alpha * (timestamp - prev_timestamp);
  cv::Mat inter_frame;
  prev_frame.copyTo(inter_frame);

  for (uint i = 0; i < next_points.size(); i++)
  {
    cv::Point2f inter_point =
        prev_points[i] + (next_points[i] - prev_points[i]) * alpha;
    // check if in bounds
    if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
        inter_point.x < 0 || inter_point.y < 0)
    {
      continue;
    }
    inter_frame.at<uchar>(inter_point) = prev_frame.at<uchar>(prev_points[i]);
    // time.at<uchar>(inter_point) = (int) (alpha * 127.0);
  }

  cv::Mat darker, lighter, darker_binary, lighter_binary;

  cv::subtract(prev_inter_frame, inter_frame, darker);
  cv::subtract(inter_frame, prev_inter_frame, lighter);

  cv::threshold(lighter, lighter, c_pos_, 255, cv::THRESH_BINARY);
  cv::threshold(darker, darker, c_neg_, 255, cv::THRESH_BINARY);
  packEvents(lighter, darker, current_timestamp, events_);

  prev_inter_frame = inter_frame;
}

return events_;
}

std::vector<cv::Mat> &SparseInterpolatedEventSimulator::getEventFrame(
    const cv::Mat prev_frame, const cv::Mat frame, int &num_frames,
    const bool getFlowFrame,
    cv::Mat &flowFrame)
{
  num_frames = 1;
  cv::Mat mask;
  cv::Mat tempFrame;
  std::vector<uchar> status;

  cv::absdiff(prev_frame, frame, mask);
  cv::threshold(mask, mask, c_pos_, 255, cv::THRESH_BINARY);

  std::vector<cv::Point2f> prev_points;
  cv::findNonZero(mask, prev_points);

  std::vector<cv::Point2f> next_points =
      optical_flow_calculator_->calculateFlow(prev_frame, frame, prev_points, status);
  cv::Mat prev_inter_frame = prev_frame;

  //void cv::arrowedLine 	(InputOutputArray img,
  //    Point 	pt1,
  //    Point 	pt2,
  //    const Scalar & 	color,
  //    int 	thickness = 1,
  //    int 	line_type = 8,
  //    int 	shift = 0,
  //    double 	tipLength = 0.1 )

  // draw flow frame
  if (getFlowFrame)
  {
    tempFrame = cv::Mat::zeros(prev_frame.size(), CV_8UC3);
    for (int i = 0; i < (int)next_points.size(); i += 5)
    {
      if (!status[i])
        continue;
      cv::arrowedLine(tempFrame, prev_points[i], next_points[i], cv::Scalar(0, 255, 0));
      // cv::circle(tempFrame, next_points[i], 2, cv::Scalar(0,255,0), -1);
      // cv::line(tempFrame, prev_points[i], next_points[i], cv::Scalar(0,255,0));
    }
    flowFrame = tempFrame;
    // std::cout<<"tempFrame.size(): "<<tempFrame.size()<<"\n";
  }

  cv::Mat darker_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  cv::Mat lighter_frame = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  int x_res = mask.cols;
  int y_res = mask.rows;

  for (int j = 0; j < num_inter_frames_ + 2; j++)
  {
    float alpha =
        static_cast<double>(j) / static_cast<double>(num_inter_frames_ + 1);
    cv::Mat inter_frame;
    prev_frame.copyTo(inter_frame);

    for (uint i = 0; i < next_points.size(); i++)
    {
      cv::Point2f inter_point =
          prev_points[i] + (next_points[i] - prev_points[i]) * alpha;
      // check if in bounds
      if (inter_point.x > (x_res - 1) || inter_point.y > (y_res - 1) ||
          inter_point.x < 0 || inter_point.y < 0)
      {
        continue;
      }
      inter_frame.at<uchar>(inter_point) = prev_frame.at<uchar>(prev_points[i]);
    }

    cv::Mat darker, lighter, darker_binary, lighter_binary;

    cv::subtract(prev_inter_frame, inter_frame, darker);
    cv::subtract(inter_frame, prev_inter_frame, lighter);

    cv::add(darker, darker_frame, darker_frame);
    cv::add(lighter, lighter_frame, lighter_frame);

    cv::threshold(lighter_frame, lighter_frame, c_pos_, 255, cv::THRESH_BINARY);
    cv::threshold(darker_frame, darker_frame, c_neg_, 255, cv::THRESH_BINARY);
    prev_inter_frame = inter_frame;
  }

  out_frames_.resize(num_frames);

  std::vector<cv::Mat> channels;
  cv::Mat zeros = cv::Mat::zeros(prev_frame.size(), CV_8UC1);
  channels.push_back(lighter_frame);
  channels.push_back(zeros);
  channels.push_back(darker_frame);
  merge(channels, out_frames_.at(0));

  return out_frames_;
}
*/