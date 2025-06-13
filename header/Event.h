#ifndef EVENT_H
#define EVENT_H

//#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>

struct ImuData{               //保留功能，但沒用到
  double time_stamp;
  cv::Vec3d linear_acceleration;
  cv::Vec3d angular_velocity;
};

struct ImageData{             //保留功能，但沒用到 
  double time_stamp;
  cv::Mat image;
  uint32_t seq;
};

struct CameraParam
{
  float fx, fy;               // focus
  float cx, cy;               // center
  float rd1, rd2;             // radial distortion
};

struct Event
{
  Event(int in_x, int in_y, float in_timestamp, bool in_polarity)
      : x{in_x}, y{in_y}, timestamp{in_timestamp}, polarity{in_polarity} {}
  int x;
  int y;
  float timestamp;            // 控制在整數三位, 小數點四位(float精度為7位). unit: Second
  bool polarity;
};

/**
 * @brief Helper function to pack data into an event struct.
 *
 * @param lighter_events cv::Mat containing pixel which got brighter
 * @param darker_events cv::Mat containing pixel which got darker
 * @param output Vector with events
 * @param timestamp Time stamp of the events
 */
void packEvents(const cv::Mat &lighter_events, const cv::Mat &darker_events,
                const float &timestamp, std::vector<Event> &output);

class EventBundle
{
public:
  EventBundle();
  EventBundle(const EventBundle &temp);
  ~EventBundle();

  /**
   * @brief 清除所有資料
   */
  void Clear();

  /**
   * @brief 複製時間、極性、角度資訊，並設定尺寸
   * 但不複製座標點(x-y, coord-2d, coord-3d)，由於座標點會再經過校正或映射
   */
  void Copy(const EventBundle &ref);

  /**
   * @brief setting coord-2D by x-y
   */
  void SetCoord();

  /**
   * @brief setting x-y by coord-2D
   */
  void SetXY();

  /**
   * @brief 附加座標點、時間、極性資訊，並更改尺寸，但不加入角度
   */
  void Append(const EventBundle &ref);

  /**
   * @brief 刪除索引值iter的事件
   */
  void erase(const size_t iter);

  /**
   * @brief Project coord_3d into coord
   */
  void Projection(const cv::Matx33d &K);

  /**
   * @brief Backproject coord-2D into coord_3d, and calculate time_delta
   */
  void InverseProjection(const cv::Matx33d &K);

  /**
   * @brief Discriminate whether the coord is inside the image window or not, then
   * stored status in x, y and isInner.
   */
  void DiscriminateInner(const int width, const int height, const int map_sampling_rate = 1);

  // coord: 2 by N uv coord or 3 by N xyz coord
  std::vector<cv::Point2f> coord;      //變動尺寸的n-by-2矩陣   Eigen::MatrixXf
  std::vector<cv::Point3f> coord_3d;

  std::vector<float> time_delta;
  std::vector<float> time_delta_reverse;
  
  cv::Vec3d angular_velocity;
  cv::Vec3d angular_position;

  //所有實際事件數據
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> time_stamp; 
  std::vector<bool> polarity;
  std::vector<bool> isInner;
  size_t size;
};

/**
 * @brief Base class for event simulators.
 */
class EventSimulator
{
public:
  /**
   * @brief Returns the simulated events from the simulator
   *
   * @param prev_frame Frame form the previous time step
   * @param frame Frame form the current time step
   * @param prev_timestamp Time stamp of the previous time step
   * @param timestamp Time stamp of the current time step
   * @param getEventFrame 是否要取得事件影像
   * @param eventFrame 事件影像
   * @param getFlowFrame 是否要取得光流影像
   * @param flowFrame 光流影像
   */
  virtual std::vector<Event> &getEvents(const cv::Mat &prev_frame,
                                        const cv::Mat &frame,
                                        const float &prev_timestamp,
                                        const float &timestamp,
                                        const cv::Rect &roi,
                                        const bool &getEventFrame,
                                        cv::Mat &eventFrame,
                                        const bool &getFlowFrame,
                                        cv::Mat &flowFrame) = 0;

  /**
   * @brief Returns the name of the event simulator
   */
  virtual std::string getName() = 0;

  /**
   * @brief Sets the frame size
   *
   * @param frame_size Size of the frame
   */
  virtual void setup(cv::Size frame_size) { frame_size_ = frame_size; }

protected:
  /// Size of the frame
  cv::Size frame_size_;

  /// Vector containing the accumulated event frames
  // 注意: 直接累積為一張影像，out_frames.size()一直為1 !
  // num_frames在Player.h中表示影片的總畫面數量。但Event.h中原作者將num_frames統一設為1，並且out_frames_.resize(num_frames)
  // 原作者可能原本打算將差值的每一張畫面獨立輸出，但後來改成直接累積為一張影像，於是out_frames_.size()一直為1
  // std::vector<cv::Mat> out_frames_;     //改在getEvent()時由&EventFrame輸出單張累積影像

  /// Vector containing the events on calculate with one frame
  std::vector<Event> events_;
};

#endif