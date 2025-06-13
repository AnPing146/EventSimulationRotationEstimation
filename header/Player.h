#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include "Event.h"
#include "System.h"
#include "numerics.h"

/**
 * @brief Abstract class defining the interface for video loader classes.
 */
class VideoLoader
{
public:
  /**
   * @brief Returns the file name of the video
   *
   * @return The file name of the video */
  std::string getFileName();

  /**
   * @brief Loads a video
   *
   * @param path Path to the video file
   * @param height Height of the video
   * @param width Width of the video
   */
  virtual void load(const std::string path, const int height = 0, const int width = 0) = 0;

  /**
   * @brief Release video resources
   */
  virtual void release() = 0;

  /**
   * @brief Returns the number of frames in the video
   *
   * @return Number of frames
   */
  std::size_t getNumFrames() { return num_frames_; }

  /**
   * @brief Returns the frame rate of the video
   *
   * @return the frame rate of the video
   */
  virtual int getFrameRate() = 0;

  /**
   * @brief Returns the height of the video
   *
   * @return The height of the video
   */
  virtual int getFrameHeight() = 0;

  /**
   * @brief Returns the width of the video
   *
   * @return The width of the video
   */
  virtual int getFrameWidth() = 0;

  /**
   * @brief Returns frame number @p index of the video
   *
   * @param index The frame index
   */
  cv::Mat getFrame(const int index) const { return frame_buffer_[index]; }

  /**
   * @brief Returns the timestamp of frame @param index
   */
  float getTimestamp(const int index) const { return timestamps_[index]; }

protected:
  /// Frame buffer as vector of cv::Mat
  std::vector<cv::Mat> frame_buffer_;

  /// Number of frames of the video
  std::size_t num_frames_;

  /// Frame rate of the video
  int frame_rate_;

  // timestamps of video frames
  std::vector<float> timestamps_;

  /// Path to the video file
  std::string path_;

  /// X resolution
  int res_x_;

  /// Y resolution
  int res_y_;
};

/**
 * @brief Class to load videos with OpenCV
 */
class OpenCVLoader : public VideoLoader
{
public:
  /**
   * @brief Loads a video
   *
   * @param path Path to the video file
   * @param height Height of the video
   * @param width Width of the video
   */
  void load(const std::string path, const int height = 0, const int width = 0) override;

  /**
   * @brief Clears the frame buffer
   */
  void release() override;

  int getFrameRate(void) override { return frame_rate_; }

  int getFrameHeight() override { return res_y_; }

  int getFrameWidth() override { return res_x_; }
};

/**
 * @brief Abstract class defining the interface for video player classes.
 */
class VideoPlayer
{
public:
  /**
   * @brief Constructor initializing the event simulator, the ROI and the loader
   *
   * @param event_simulator The event simulator used by the video player
   * @param res_x X resolution    //沒用到, loader時已經設定解析度了
   * @param res_y Y resolution
   */
  VideoPlayer(std::shared_ptr<EventSimulator> event_simulator, int res_x = 0,
              int res_y = 0, std::shared_ptr<System> system = nullptr);

  /**
   * @brief Loads a video, sets up the event simulator and runs the simulation
   *        @p repeats number of times.
   *
   * @param path Path to the video
   * @param height Height of the video frames
   * @param width Width of the video frames
   * @param repeats Number of times the simulation should be run
   * @param event_statistics Flag indicating if event statistics should be
   * calculated
   * @param record_video Flag indicating if a video should be recorded
   * @param saveFlowFrame 是否保留光流影像為影片
   */
  void simulate(const std::string path, const int height = 0,
                const int width = 0, const int repeats = 1,
                const bool event_statistics = false,
                const bool record_video = false,
                const bool saveFlowFrame = false);

  /**
   * @brief Run a timed simulation. No event statistics will be calculated
   *        and not output video recorded.
   *
   * @param path Path to the video
   * @param height Height of the video frames
   * @param width Width of the video frames
   * @param repeats Number of times the simulation should be run
   * @param num_frames Number of frames in the video
   * @return The average run time per frame [ms]
   */
  double simulateTimed(const std::string path, const int height = 0,
                       const int width = 0, const int repeats = 1,
                       const int num_frames = 0);

  /**
   * @brief Simulates the event for a single frame and saves the result. 尚未實作事件輸出功能則
   *
   * @param path Path to the video
   * @param height Height of the video frames
   * @param width Width of the video frames
   * @param frame_index Index of the frame
   * @param saveFlowFrame 是否保留光流影像為影片
   */
  void saveSingleFrame(const std::string path, const int height = 0,
                       const int width = 0, const int frame_index = 1,
                       const bool saveFlowFrame = false);

  /**
   * @brief Save record angular data
   */
  void saveRecordAngular();

  /**
   * @brief Set the region of interest (ROI)
   *
   * @param roi Region of interest
   */
  void setROI(const cv::Rect roi) { roi_ = roi; }

  /**
   * @brief 手動設定system
   */
  void setSystem(std::shared_ptr<System> system) { system_ = system; }

protected:
  /**
   * @brief Returns the next frame from the video loader
   */
  cv::Mat getNextFrame();

  /**
   * @brief return the timestamp of current frame in milliseconds
   */
  float getTimestamp();

  /**
   * @brief Runs the simulation @p repeats number of times. If @p num_frames is
   * not set it will be determined by the video loader. Event statistics of the
   * simulation are recorded if @p event_statistics is set. A video is recorded
   * if @p record_video is set.
   * 會傳入roi_給event_simulator_->getEvents(...)做事件roi
   * 注意: 影片與光流影片的檔案處理在VideoPlayer類別的子類別成員函式中被實作!!
   *
   * @param repeats Number of simulations to run
   * @param num_frames Number of frames the video has
   * @param event_statistics Flag indicating if event statistics should be
   * calculated
   * @param record_video Flag indicating if a video should be recorded
   * @param saveFlowFrame 是否保留光流影像為影片
   */
  virtual void loopSimulation(const int repeats, int num_frames,
                              const bool event_statistics,
                              const bool record_video,
                              const bool saveFlowFrame) = 0;

  /**
   * @brief Returns the frame size of the video.
   */
  cv::Size getFrameSize() { return loader_->getFrame(0).size(); }

  /**
   * @brief Sets up the event simulator.
   */
  void setupEventSimulator() { event_simulator_->setup(getFrameSize()); }

  /**
   * @brief 檢查是否有global alignment rotation estimation
   */
  bool hasSystem() { return system_ != nullptr; }

  /// Event simulator
  std::shared_ptr<EventSimulator> event_simulator_;

  // global alignment rotation estimation
  std::shared_ptr<System> system_;

  /// Number of the current frame
  std::size_t current_frame_;

  /// Region of intererst
  cv::Rect roi_;

  /// Video loader
  std::shared_ptr<VideoLoader> loader_;
};

/**
 * @brief Video player using OpenCV.
 */
class OpenCVPlayer : public VideoPlayer
{
public:
  /**
   * @brief Constructor initializing the base class, the wait time and pause
   *        flag.
   *
   * @param event_simulator Event simulator
   * @param wait_time_ms Waiting time for observe each frame [ms]
   * @param res_x X resolution        //沒用到, loader時已經設定解析度了
   * @param res_y Y resolution
   */
  OpenCVPlayer(std::shared_ptr<EventSimulator> event_simulator,
               const int wait_time_ms, const int res_x = 0,
               const int res_y = 0, std::shared_ptr<System> system = nullptr);

  /**
   * @brief Sets the event simulator
   *
   * @param event_simulator Event simulator
   */
  void setEventSimulator(std::shared_ptr<EventSimulator> event_simulator)
  {
    event_simulator_ = event_simulator;
  }

private:
  /**
   * 計算模擬事件，如果有實例化system_，則會將事件傳入system_進行全域對齊
   */
  void loopSimulation(const int repeats, int num_frames,
                      const bool event_statistics,
                      const bool record_video,
                      const bool saveFlowFrame) override;

  /// Wait time [ms]
  int wait_time_ms_;
};

/**
 * @brief Video streamer class
 */
class VideoStreamer
{
public:
  /**
   * @brief Constructor initializing the event simulaor and disabling the ROI.
   *
   * @param event_simulator Event simulator
   */
  VideoStreamer(std::shared_ptr<EventSimulator> event_simulator,
                std::shared_ptr<System> system = nullptr);

  /**
   * @brief Simulate events given a video stream source.
   *
   * @param source_index Index of the video stream source
   */
  void simulateFromStream(const int source_index);

protected:
  /**
   * @brief 檢查是否有global alignment rotation estimation
   */
  bool hasSystem() { return system_ != nullptr; }

  /// Event simulator
  std::shared_ptr<EventSimulator> event_simulator_;

  // global alignment rotation estimation
  std::shared_ptr<System> system_;

  /// Region of interest
  cv::Rect roi_;

  /// Video loader
  std::shared_ptr<VideoLoader> loader_;
};

/**
 * @brief 事件播放器。原本用於以事件資料集運行global alignment rotate estimation，後來新增功能
 * 也可用於比較事件資料集與模擬事件的差異
 */
class EventDataPlayer
{
public:
  /**
   * @brief 建構式並初始化system成員
   */
  EventDataPlayer(const cv::Rect &roi) : roi_{roi} {}

  /**
   * @brief 讀取資料集事件. 讀取成功回傳Ture
   */
  bool readEventFile(const std::string &filePath);

  /**
   * @brief 讀取images.txt的標準影像時間戳，回傳總時間戳數量
   */
  int readImagesTimestamp(const std::string &filePath);

  /**
   * @brief 取出index的標準影像時間戳
   */
  float getImageTimestamp(const int &index);

  /**
   * @brief 儲存單張累積事件影像。 可用於比較與模擬事件的差異
   */
  cv::Mat getSingleAccmulateFrame(const int &row, const int &col, const float &timestamp_prev, const float &timestamp_current);

  /**
   * @brief 取得時間間隔內的事件數量
   */
  double getInterval(const float &timestamp_prev, const float &timestamp_current);

  /**
   * @brief 輸出index的事件資訊(用於檢查事件)
   */
  void printIndexEvent(const uint32_t &index);

  /**
   * @brief 輸出事件到global alignment roatate estimation system
   */
  void publishData(const uint16_t &bunchSize);

  /**
   * @brief 設定roi
   */
  void setRoi(const cv::Rect roi) { roi_ = roi; }

  /**
   * @brief 初始化system_
   */
  void setSystem(std::shared_ptr<System> system) { system_ = system; }

  // void VideoPlayer::saveRecordAngular()

protected:
  uint32_t size;
  std::vector<Event> eventData; // timestamp轉型為float!
  std::vector<float> imageTimestamp;
  cv::Rect roi_;

  // global alignment rotation estimation
  std::shared_ptr<System> system_;
};