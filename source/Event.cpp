#include "Event.h"
#include "numerics.h"

// #include <Eigen/Core>
#include <opencv2/core.hpp>

#include <string>

void packEvents(const cv::Mat &lighter_events, const cv::Mat &darker_events,
                const float &timestamp, std::vector<Event> &output)
{
  std::vector<cv::Point2d> neg_polarity_events;
  std::vector<cv::Point2d> pos_polarity_events;

  cv::findNonZero(lighter_events, pos_polarity_events);
  cv::findNonZero(darker_events, neg_polarity_events);

  output.reserve(output.size() + neg_polarity_events.size() +
                 pos_polarity_events.size());

  for (const auto &point : neg_polarity_events)
  {
    output.emplace_back(point.x, point.y, timestamp, false);
  }
  for (const auto &point : pos_polarity_events)
  {
    output.emplace_back(point.x, point.y, timestamp, true);
  }

  /*
    if (roi == cv::Rect(0, 0, 0, 0))
    {
      for (const auto &point : neg_polarity_events)
      {
        output.emplace_back(point.x, point.y, timestamp, false);
      }
      for (const auto &point : pos_polarity_events)
      {
        output.emplace_back(point.x, point.y, timestamp, true);
      }
    }
    else
    {
      for (const auto &point : neg_polarity_events)
      {
        if (point.x < roi.x || point.x > (roi.x + roi.width - 1) || point.y < roi.y || point.y > (roi.y + roi.height - 1))
          continue;
        output.emplace_back(point.x, point.y, timestamp, false);
      }
      for (const auto &point : pos_polarity_events)
      {
        if (point.x < roi.x || point.x > (roi.x + roi.width - 1) || point.y < roi.y || point.y > (roi.y + roi.height - 1))
          continue;
        output.emplace_back(point.x, point.y, timestamp, true);
      }
    }
  */
}

EventBundle::EventBundle()
{
  size = 0;
  coord.resize(size);
  coord_3d.resize(size);
}

EventBundle::EventBundle(const EventBundle &temp)
{
  coord = temp.coord;
  coord_3d = temp.coord_3d;

  angular_velocity = temp.angular_velocity;
  angular_position = temp.angular_position;

  time_delta = temp.time_delta;
  time_delta_reverse = temp.time_delta_reverse;

  time_stamp = temp.time_stamp;
  polarity = temp.polarity;
  x = temp.x;
  y = temp.y;
  isInner = temp.isInner;
  size = temp.size;
}

EventBundle::~EventBundle()
{
}

void EventBundle::Clear()
{
  size = 0;

  coord.resize(size);
  coord_3d.resize(size);

  time_delta.resize(size);
  time_delta_reverse.resize(size);

  angular_velocity = cv::Vec3d::all(0.0);
  angular_position = cv::Vec3d::all(0.0);

  time_stamp.clear();
  polarity.clear();
  x.clear();
  y.clear();
  isInner.clear();
}

void EventBundle::Copy(const EventBundle &ref)
{
  time_delta = ref.time_delta;
  time_delta_reverse = ref.time_delta_reverse;

  time_stamp = ref.time_stamp;
  polarity = ref.polarity;
  size = ref.size;

  angular_velocity = ref.angular_velocity;
  angular_position = ref.angular_position;

  coord.resize(size);
  coord_3d.resize(size);
}

void EventBundle::SetCoord()
{
  size = x.size();
  if (size != 0)
  {
    coord.resize(size);
    cv::Point2f *coord_ptr = coord.data();
    const float *x_ptr = x.data();
    const float *y_ptr = y.data();

    for (std::size_t i = 0; i < size; i++)
    {
      *(coord_ptr + i) = cv::Point2f(*(x_ptr + i), *(y_ptr + i));
    }
  }
}

void EventBundle::SetXY()
{
  size = coord.size();
  x.resize(size);
  y.resize(size);

  const cv::Point2f *coord_ptr = coord.data();
  float *x_ptr = x.data();
  float *y_ptr = y.data();

  for (std::size_t i = 0; i < size; ++i)
  {
    x_ptr[i] = coord_ptr[i].x;
    y_ptr[i] = coord_ptr[i].y;
  }
}

void EventBundle::Append(const EventBundle &ref)
{
  size += ref.size;
  // 統一預先分配記憶體空間
  coord.reserve(coord.size() + ref.coord.size());
  coord_3d.reserve(coord_3d.size() + ref.coord_3d.size());
  time_delta.reserve(time_delta.size() + ref.time_delta.size());
  time_delta_reverse.reserve(time_delta_reverse.size() + ref.time_delta_reverse.size());
  time_stamp.reserve(time_stamp.size() + ref.time_stamp.size());
  polarity.reserve(polarity.size() + ref.polarity.size());

  // 再加入資料
  coord.insert(coord.end(), ref.coord.begin(), ref.coord.end());
  coord_3d.insert(coord_3d.end(), ref.coord_3d.begin(), ref.coord_3d.end());
  time_delta.insert(time_delta.end(), ref.time_delta.begin(), ref.time_delta.end());
  time_delta_reverse.insert(time_delta_reverse.end(), ref.time_delta_reverse.begin(), ref.time_delta_reverse.end());
  time_stamp.insert(time_stamp.end(), ref.time_stamp.begin(), ref.time_stamp.end());
  polarity.insert(polarity.end(), ref.polarity.begin(), ref.polarity.end());
}

void EventBundle::erase(const size_t iter)
{
  time_stamp.erase(time_stamp.begin() + iter);
  polarity.erase(polarity.begin() + iter);
  x.erase(x.begin() + iter);
  y.erase(y.begin() + iter);
}

void EventBundle::Projection(const cv::Matx33d &K)
{
  coord.resize(size);

  float fx = K(0, 0), fy = K(1, 1);
  float cx = K(0, 2), cy = K(1, 2);

  cv::Point3f *coord_3d_ptr = coord_3d.data();
  cv::Point2f *coord_2d_ptr = coord.data();

  for (std::size_t i = 0; i < size; ++i)
  {
    (coord_2d_ptr + i)->x = (coord_3d_ptr + i)->x * fx / (coord_3d_ptr + i)->z + cx;
    (coord_2d_ptr + i)->y = (coord_3d_ptr + i)->y * fy / (coord_3d_ptr + i)->z + cy;
  }
}

void EventBundle::InverseProjection(const cv::Matx33d &K)
{
  time_delta.resize(size);
  time_delta_reverse.resize(size);
  coord_3d.resize(size);

  const float t_front = time_stamp.front();
  const float t_back = time_stamp.back();

  const float *time_ptr = time_stamp.data();
  float *delta_ptr = time_delta.data();
  float *delta_rev_ptr = time_delta_reverse.data();

  const cv::Point2f *coord_ptr = coord.data();
  cv::Point3f *coord_3d_ptr = coord_3d.data();

  const float fx = K(0, 0);
  const float fy = K(1, 1);
  const float cx = K(0, 2);
  const float cy = K(1, 2);

  for (std::size_t i = 0; i < size; ++i)
  {
    *(delta_ptr + i) = t_front - *(time_ptr + i);
    *(delta_rev_ptr + i) = t_back - *(time_ptr + i);

    float x_norm = ((coord_ptr + i)->x - cx) / fx;
    float y_norm = ((coord_ptr + i)->y - cy) / fy;
    *(coord_3d_ptr + i) = cv::Point3f(x_norm, y_norm, 1.0f);
  }
}

// Discriminate whether the coord is inside the image window or not
void EventBundle::DiscriminateInner(const int width, const int height, const int map_sampling_rate) // map_sampling_rate = 1
{
  // TODO: xy與coord內容可能被修改過，需先計算新size，而不是直接使用舊size!!
  // 加入 uint32_t getSize()
  isInner.resize(size);
  if (x.size() != size)
  {
    SetXY();
  }
  for (uint32_t pts_iter = 0; pts_iter < size; pts_iter++)
  {
    if (x[pts_iter] <= 0 || x[pts_iter] >= width || y[pts_iter] <= 0 || y[pts_iter] >= height || pts_iter % map_sampling_rate != 0)
    {
      isInner[pts_iter] = false;
    }
    else
    {
      isInner[pts_iter] = true;
    }
  }
}

/*
void packEvents(const cv::Mat &lighter_events, const cv::Mat &darker_events,
                const float timestamp, std::vector<Event> &output) {
  std::vector<cv::Point2d> neg_polarity_events;
  std::vector<cv::Point2d> pos_polarity_events;

  cv::findNonZero(lighter_events, pos_polarity_events);
  cv::findNonZero(darker_events, neg_polarity_events);

  output.reserve(output.size() + neg_polarity_events.size() +
                 pos_polarity_events.size());

  for (const auto &point : neg_polarity_events) {
    output.emplace_back(point.x, point.y, timestamp, false);
  }
  for (const auto &point : pos_polarity_events) {
    output.emplace_back(point.x, point.y, timestamp, true);
  }
}

EventBundle::EventBundle(){
  size = 0;
  coord.resize(size, 2);
  coord_3d.resize(size, 3);
}

EventBundle::EventBundle(const EventBundle& temp){
  coord = temp.coord;
  coord_3d = temp.coord_3d;

  angular_velocity = temp.angular_velocity;
  angular_position = temp.angular_position;

  time_delta = temp.time_delta;
  time_delta_reverse = temp.time_delta_reverse;

  time_stamp = temp.time_stamp;
  polarity = temp.polarity;
  x = temp.x;
  y = temp.y;
  isInner = temp.isInner;
  size = temp.size;
}

EventBundle::~EventBundle(){
}

void EventBundle::Clear(){
  size = 0;

  coord.resize(size, 2);
  coord_3d.resize(size, 3);

  time_delta.resize(size);
  time_delta_reverse.resize(size);

  angular_velocity = Eigen::Vector3f::Zero();
  angular_position = Eigen::Vector3f::Zero();

  time_stamp.clear();
  polarity.clear();
  x.clear();
  y.clear();
  isInner.clear();
}

void EventBundle::Copy(const EventBundle &ref){
  time_delta = ref.time_delta;
  time_delta_reverse = ref.time_delta_reverse;

  time_stamp = ref.time_stamp;
  polarity = ref.polarity;
  size = ref.size;

  angular_velocity = ref.angular_velocity;
  angular_position = ref.angular_position;

  coord.resize(size,2);
  coord_3d.resize(size,3);
}

void EventBundle::SetCoord(){
  size = x.size();
  if(size != 0){
      Eigen::Map<Eigen::VectorXf> eigen_x(x.data(), size);
      Eigen::Map<Eigen::VectorXf> eigen_y(y.data(), size);
      coord = Eigen::MatrixXf(size, 2);   //改成push_back <cv::Point2f>
      coord << eigen_x, eigen_y;
  }
}

// Synchronize the coord and the x, y
void EventBundle::SetXY(){
  x = std::vector<float>(coord.col(0).data(), coord.col(0).data() + size);
  y = std::vector<float>(coord.col(1).data(), coord.col(1).data() + size);
}

void EventBundle::Append(const EventBundle &ref){

  size += ref.size;

  Eigen::MatrixXf temp_coord = coord;
  Eigen::MatrixXf temp_coord_3d = coord_3d;
  Eigen::VectorXf temp_time_delta = time_delta;
  Eigen::VectorXf temp_time_delta_reverse = time_delta_reverse;

  coord.resize(size, 2);
  coord_3d.resize(size, 3);

  time_delta.resize(size);
  time_delta_reverse.resize(size);

  coord << temp_coord, ref.coord;
  coord_3d << temp_coord_3d, ref.coord_3d;
  time_delta << temp_time_delta, ref.time_delta;
  time_delta_reverse << temp_time_delta_reverse, ref.time_delta_reverse;

  time_stamp.insert(time_stamp.end(), ref.time_stamp.begin(), ref.time_stamp.end());
  polarity.insert(polarity.end(), ref.polarity.begin(), ref.polarity.end());


}

void EventBundle::erase(size_t iter){
  time_stamp.erase(time_stamp.begin() + iter);
  polarity.erase(polarity.begin() + iter);
  x.erase(x.begin() + iter);
  y.erase(y.begin() + iter);
}

// Project coord_3d into coord
void EventBundle::Projection(Eigen::Matrix3f K){
  coord.col(0) = coord_3d.col(0).array()/coord_3d.col(2).array() * K(0, 0) + K(0, 2);
  coord.col(1) = coord_3d.col(1).array()/coord_3d.col(2).array() * K(1, 1) + K(1, 2);
  coord = coord.array().round();
}

// Backproject coord into coord_3d
void EventBundle::InverseProjection(Eigen::Matrix3f K){
  Eigen::Map<Eigen::VectorXf> time_delta_(time_stamp.data(), size);
  time_delta = (time_stamp.front()- time_delta_.array()).cast<float>();
  time_delta_reverse = (time_stamp.back()- time_delta_.array()).cast<float>();
  coord_3d.col(0) = (coord.col(0).array() - K(0, 2))/K(0, 0);
  coord_3d.col(1) = (coord.col(1).array() - K(1, 2))/K(1, 1);
  coord_3d.col(2) = Eigen::MatrixXf::Ones(size, 1);
}

// Discriminate whether the coord is inside the image window or not
void EventBundle::DiscriminateInner(int width, int height, int map_sampling_rate) // = 1
{
  //TODO: xy與coord內容可能被修改過，需先計算新size，而不是直接使用舊size!!
  //加入 uint32_t getSize()
  isInner.resize(size);
  if (x.size() != size)
  {
      SetXY();
  }
  for (uint32_t pts_iter = 0; pts_iter < size; pts_iter++)
  {
      if (x[pts_iter] <= 0 || x[pts_iter] >= width || y[pts_iter] <= 0 || y[pts_iter] >= height || pts_iter % map_sampling_rate != 0)
      {
          isInner[pts_iter] = false;
      }
      else
      {
          isInner[pts_iter] = true;
      }
  }
}
*/
