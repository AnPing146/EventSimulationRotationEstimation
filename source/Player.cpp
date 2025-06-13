#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "Player.h"
#include "Event.h"
#include "System.h"
#include "numerics.h"

std::string VideoLoader::getFileName()
{
  std::string str(path_);
  return str.substr(str.find_last_of("/\\") + 1);
}

void OpenCVLoader::load(const std::string path, const int height, const int width)
{
  path_ = path;
  cv::VideoCapture cap(path);

  if (!cap.isOpened())
  {
    throw std::runtime_error("Error reading video file. Does it exist?");
  }

  num_frames_ = cap.get(cv::CAP_PROP_FRAME_COUNT);
  frame_rate_ = cap.get(cv::CAP_PROP_FPS);

  if (height == 0)
  {
    res_y_ = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  }
  else
  {
    res_y_ = height;
  }
  if (width == 0)
  {
    res_x_ = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  }
  else
  {
    res_x_ = width;
  }

  std::cout << "Creating framebuffer of: " << num_frames_ << " n Frames"
            << std::endl;

  frame_buffer_.reserve(num_frames_);

  for (int i = 0; i < num_frames_; i++)
  {
    cv::Mat frame;
    cap >> frame;
    // Resize frame to chosen resolution if frame is bigger
    if (res_y_ > 0 && res_x_ > 0)
    {
      cv::Mat resized_frame;
      cv::resize(frame, resized_frame, cv::Size(res_x_, res_y_), cv::INTER_LINEAR);
      frame_buffer_.emplace_back(resized_frame);
      timestamps_.emplace_back(round4(cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0f));
    }
    else
    {
      frame_buffer_.emplace_back(frame);
      timestamps_.emplace_back(round4(cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0f));
    }
  }

  std::cout << "Finished creating framebuffer" << std::endl
            << std::endl;
}

void OpenCVLoader::release()
{
  frame_buffer_.clear();
}

/**
 * @brief Color to gray converstion
 *
 * @param frame Color frame in BGR
 */
cv::Mat toGray(const cv::Mat &frame)
{
  cv::Mat grey_frame;
  cv::cvtColor(frame, grey_frame, cv::COLOR_BGR2GRAY);
  return grey_frame;
}

/**
 * @brief Returns a random hex number as string.
 *
 * @param length Lenght of the hexadecimal number
 */
std::string getRandomHex(int length)
{
  std::array<char, 16> hexChar = {'0', '1', '2', '3', '4', '5', '6', '7',
                                  '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

  std::string output;
  // Loop to print N integers
  for (int i = 0; i < length; i++)
  {
    output += hexChar[rand() % 16];
  }
  return output;
}

VideoPlayer::VideoPlayer(std::shared_ptr<EventSimulator> event_simulator,
                         int res_x, int res_y, std::shared_ptr<System> system)
    : event_simulator_{event_simulator},
      current_frame_{0},
      roi_{cv::Rect(0, 0, 0, 0)},
      loader_{std::make_shared<OpenCVLoader>()},
      system_{system} {}

void VideoPlayer::simulate(const std::string path, const int height,
                           const int width, const int repeats,
                           const bool event_statistics,
                           const bool record_video,
                           const bool saveFlowFrame)
{
  loader_->load(path, height, width);

  setupEventSimulator();

  int start = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

  loopSimulation(repeats, 0, event_statistics, record_video, saveFlowFrame);

  int end = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();

  loader_->release();

  int ms = end - start;
  std::cout << std::endl
            << "Frame time of "
            << static_cast<double>(ms) / static_cast<double>(current_frame_)
            << "ms" << std::endl;
  std::cout << "Diff of " << current_frame_ << " Frames in " << ms << "ms = "
            << static_cast<double>(current_frame_) /
                   (static_cast<double>(ms) / 1000.0)
            << " FPS" << std::endl;

  // Reset counter to be ready to play again
  current_frame_ = 0;
}

double VideoPlayer::simulateTimed(const std::string path, const int height,
                                  const int width, const int repeats,
                                  const int num_frames)
{
  loader_->load(path, height, width);

  setupEventSimulator();

  int start = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

  loopSimulation(repeats, num_frames, false, false, false);

  int end = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();

  loader_->release();

  int ms = (end - start);
  double frametime =
      static_cast<double>(ms) / static_cast<double>(current_frame_ * repeats);
  std::cout << std::endl
            << "Frame time of "
            << static_cast<double>(ms) / static_cast<double>(current_frame_)
            << "ms" << std::endl;
  std::cout << "Diff of " << current_frame_ << " Frames in " << ms << "ms = "
            << static_cast<double>(current_frame_) /
                   (static_cast<double>(ms) / 1000.0)
            << " FPS" << std::endl;

  // Reset counter to be ready to play again
  current_frame_ = 0;

  return frametime;
}

void VideoPlayer::saveSingleFrame(const std::string path, const int height,
                                  const int width, const int frame_index,
                                  const bool saveFlowFrame)
{
  if (loader_->getNumFrames() < 3 || frame_index < 3)
  {
    std::cout << "ERROR: Frame index must be at least 3 and video must contain "
                 "3 frames"
              << std::endl;
  }

  loader_->load(path, height, width);

  setupEventSimulator();

  cv::Mat first = toGray(loader_->getFrame(frame_index - 2));
  cv::Mat second = toGray(loader_->getFrame(frame_index - 1));
  cv::Mat flowFrame[2] = {cv::Mat(loader_->getFrameHeight(), loader_->getFrameWidth(), CV_8UC3),
                          cv::Mat(loader_->getFrameHeight(), loader_->getFrameWidth(), CV_8UC3)}; //////////////////////
  cv::Mat result;
  int num_frames = 1;

  int frame_rate = loader_->getFrameRate();
  float time_per_frame = round4(1 / static_cast<float>(frame_rate));
  float timestamp_previous2 = round4((frame_index - 2) * time_per_frame);
  float timestamp_previous = timestamp_previous2 + time_per_frame;
  float timestamp_current = timestamp_previous + time_per_frame;

  // 第一次計算影像時暫時用result存放，第二次計算才是我們要的結果影像
  //// 事件輸出功能則尚未實作!
  std::vector<Event> &previous_events = event_simulator_->getEvents(first, second,
                                                                    timestamp_previous2, timestamp_previous,
                                                                    roi_, true, result, saveFlowFrame, flowFrame[0]);
  std::vector<Event> &current_events = event_simulator_->getEvents(second, toGray(loader_->getFrame(frame_index)),
                                                                   timestamp_previous, timestamp_current,
                                                                   roi_, true, result, saveFlowFrame, flowFrame[1]);

  /*
  event_simulator_->getEventFrame(first, second, num_frames, saveFlowFrame, flowFrame[0]);
  auto results = event_simulator_->getEventFrame(
      second, toGray(loader_->getFrame(frame_index)), num_frames, saveFlowFrame, flowFrame[1]);
  */

  std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
  std::string::size_type const p(base_filename.find_last_of('.'));
  std::string file_without_extension = base_filename.substr(0, p);
  std::string output_path = file_without_extension;

  for (int i = 0; i < num_frames; i++)
  {
    std::string random_id = getRandomHex(6);
    std::string file_name = std::to_string(frame_index) + "_" +
                            event_simulator_->getName() + ".png";
    if (roi_.width > 0 && roi_.height > 0)
    {
      cv::imwrite(output_path + "_" + file_name, result(roi_));
    }
    else
    {
      cv::imwrite(output_path + "_" + file_name, result);
    }
    std::cout << "Saved frame " << frame_index << " to "
              << output_path + "_" + file_name << std::endl;
  }
}

void VideoPlayer::saveRecordAngular()
{
  if (system_ == nullptr)
  {
    throw std::runtime_error("No allocated System exist! Did you forget declare System?\n");
  }

  std::vector<float> timestamp = system_->GetRecordTimestamp();
  std::vector<cv::Vec3d> velocity = system_->GetRecordVelocity();
  std::vector<cv::Vec3d> position = system_->GetRecordPosition();
  uint size = timestamp.size();

  // 取得當前時刻，生成檔案名稱
  auto now = std::chrono::system_clock::now();
  time_t now_time = std::chrono::system_clock::to_time_t(now);
  tm *local_time = localtime(&now_time);

  std::stringstream filename;
  filename << "record_angular_"
           << std::put_time(local_time, "%Y%m%d_%H%M%S") // 格式化時間
           << ".txt";

  std::ofstream outfile(filename.str());
  if (!outfile.is_open())
  {
    throw std::runtime_error("Error: Could not open file for writing.\n");
  }

  for (uint i = 0; i < size; i++)
  {
    outfile << timestamp[i] << " ";
    for (uint j = 0; j < 3; j++)
    {
      outfile << velocity[i][j] << " ";
    }
    for (uint j = 0; j < 3; j++)
    {
      outfile << position[i][j] << " ";
    }

    outfile << "\n";
  }

  outfile.close();
  std::cout << "Record data written to file: " << filename.str() << "\n";
}

cv::Mat VideoPlayer::getNextFrame()
{
  current_frame_++;
  if (loader_->getNumFrames() == 0)
  {
    throw std::runtime_error(
        "No frames in video! Did you forget to call loader->load(path)?");
  }
  return loader_->getFrame(current_frame_ % loader_->getNumFrames());
}

float VideoPlayer::getTimestamp()
{
  if (loader_->getNumFrames() == 0)
  {
    throw std::runtime_error(
        "No frames in video! Did you forget to call loader->load(path)?");
  }
  return loader_->getTimestamp(current_frame_ % loader_->getNumFrames());
}

OpenCVPlayer::OpenCVPlayer(std::shared_ptr<EventSimulator> event_simulator,
                           const int wait_time_ms, const int res_x,
                           const int res_y, std::shared_ptr<System> system)
    : VideoPlayer(event_simulator, res_x, res_y, system), wait_time_ms_{wait_time_ms} {}

void OpenCVPlayer::loopSimulation(const int repeats, int num_frames,
                                  const bool event_statistics,
                                  const bool record_video,
                                  const bool saveFlowFrame)
{
  auto frame_rate = loader_->getFrameRate();
  float time_per_frame = round4(1 / static_cast<float>(frame_rate));
  if (num_frames == 0)
  {
    num_frames = loader_->getNumFrames();
  }
  double seconds = num_frames * time_per_frame;
  float timestamp_previous, timestamp_current;
  cv::Mat frame, out_frame;
  cv::Mat flowFrame(loader_->getFrameHeight(), loader_->getFrameWidth(), CV_8UC3); ///////////////////////
  cv::Mat prev_frame = toGray(getNextFrame());
  timestamp_previous = getTimestamp();

  auto rows = loader_->getFrameHeight();
  auto cols = loader_->getFrameWidth();
  cv::Mat total_events_per_pixel = cv::Mat::zeros(rows, cols, CV_64F);
  cv::Mat pos_events_per_pixel = cv::Mat::zeros(rows, cols, CV_64F);
  cv::Mat neg_events_per_pixel = cv::Mat::zeros(rows, cols, CV_64F);

  std::cout << "Height: " << rows << ", width: " << cols << std::endl;
  std::cout << "Framerate: " << frame_rate << std::endl;
  std::cout << "Number of frames: " << num_frames << std::endl;
  std::cout << "Time per frame: " << time_per_frame << "s" << std::endl;
  std::cout << "Video duration: " << num_frames / frame_rate << "s"
            << std::endl;
  std::cout << "Waiting each frame for " << wait_time_ms_ << "ms" << std::endl;

  cv::VideoWriter video_capture, flow_capture;
  if (record_video)
  {
    const auto simulator_name = event_simulator_->getName();
    auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    video_capture.open(simulator_name + "_video.mp4", fourcc, frame_rate,
                       cv::Size(cols, rows));
    if (saveFlowFrame)
    { ///////////////////////////
      flow_capture.open(simulator_name + "_flow.mp4", fourcc, frame_rate,
                        cv::Size(cols, rows));
      // std::cout<<"flowFrame.type: "<<flowFrame.type()<<", depth:"<<flowFrame.depth()<<"\n";
      // std::cout<<"prev_frame.type: "<<prev_frame.type()<<", depth:"<<prev_frame.depth()<<"\n\n";
    }
  }

  bool should_quit = false;
  while (!should_quit) // main loop calculate events
  {
    if ((repeats > 1 && repeats <= (current_frame_ / num_frames)) ||
        (repeats == 1 && current_frame_ >= num_frames))
    {
      break;
    }

    frame = toGray(getNextFrame());
    timestamp_current = getTimestamp();
    // int num_frames; // 原作者在VideoLoader類別中表示影片的總畫面數量，但在這邊表示為一次事件計算的畫面數量

    std::vector<Event> &events = event_simulator_->getEvents(prev_frame, frame, timestamp_previous, timestamp_current,
                                                             roi_, true, out_frame, saveFlowFrame, flowFrame);
    /*
    auto out_frames = event_simulator_->getEventFrame(prev_frame, frame, num_frames, saveFlowFrame, flowFrame); //////////////////////////////
    */

    // global alignment rotation estimate
    if (hasSystem())
    {
      system_->BindEvent(events);
    }

    if (record_video && video_capture.isOpened())
    {
      // for (const auto &frame : out_frame){    //原先的影像向量，被改為單張的累積影像
      video_capture << frame;
      //}
    }
    if (record_video && flow_capture.isOpened())
    {
      flowFrame.convertTo(flowFrame, CV_8U, 255.0);
      flow_capture << flowFrame;
      cv::namedWindow("flow"); // 如果不需要顯示光流影像，請註解這三行
      cv::imshow("flow", flowFrame);
      cv::waitKey(1);
    }

    if (event_statistics)
    {
      // double prev_timestamp = (current_frame_ - 1) * time_per_frame;
      // double timestamp = current_frame_ * time_per_frame;
      //  auto events = event_simulator_->getEvents(prev_frame, frame, prev_timestamp, timestamp, num_frames);

      double *total_ptr = total_events_per_pixel.ptr<double>();
      double *pos_ptr = pos_events_per_pixel.ptr<double>();
      double *neg_ptr = neg_events_per_pixel.ptr<double>();

      int width = total_events_per_pixel.cols;

      for (const auto &event : events)
      {
        // total_events_per_pixel.at<double>(event.y, event.x) += 1.0;
        int index = event.y * width + event.x;

        *(total_ptr + index) += 1.0;

        if (event.polarity)
        {
          // pos_events_per_pixel.at<double>(event.y, event.x) += 1.0;
          *(pos_ptr + index) += 1.0;
        }
        else
        {
          // neg_events_per_pixel.at<double>(event.y, event.x) += 1.0;
          *(neg_ptr + index) += 1.0;
        }
      }
    }

    // 延遲顯示時間，以便觀察事件影像並儲存
    if (wait_time_ms_ > 0)
    {
      // for (int i = 0; i < num_frames; i++){
      cv::imshow("OpenCVPlayer", out_frame); // out_frames.at(i)
      char c = static_cast<char>(cv::waitKey(wait_time_ms_));
      if (c == 27)
      { // esc. to quit
        should_quit = true;
      }
      if (c == 115)
      { // s for save
        std::stringstream ss;
        ss << "../res/export_frames/" << loader_->getFileName() << "_" << static_cast<int>(getTimestamp() * 1000) << "ms.png";
        std::cout << "Saving frame to file: " << ss.str();
        cv::imwrite(ss.str(), out_frame);
      }
      //}
      if (should_quit)
      {
        break;
      }
    }
    prev_frame = frame;
    timestamp_previous = timestamp_current;
  }
  cv::destroyAllWindows();

  if (event_statistics)
  {
    std::ofstream myfile;
    const auto simulator_name = event_simulator_->getName();
    myfile.open(simulator_name + "_events_per_pixel.csv");
    myfile << "y,x,events_per_pixel,events_per_pixel_per_second\n";
    for (std::size_t y = 0; y < total_events_per_pixel.rows; y++)
    {
      for (std::size_t x = 0; x < total_events_per_pixel.cols; x++)
      {
        myfile << y << "," << x << ","
               << total_events_per_pixel.at<double>(y, x) << ","
               << total_events_per_pixel.at<double>(y, x) / seconds << '\n';
      }
    }
    myfile.close();

    double min, max;
    cv::minMaxIdx(total_events_per_pixel, &min, &max);
    std::cout << "Min: " << min << ", max: " << max << std::endl;
    cv::Mat output;
    cv::normalize(total_events_per_pixel, output, 0.0, 255.0, cv::NORM_MINMAX,
                  CV_8U);
    // events_per_pixel.convertTo(output, CV_8U, 1.0);

    cv::imwrite(simulator_name + "events_per_pixel_from_sim.png", output);
    cv::FileStorage fs(simulator_name + "events_per_pixel_from_sim.json",
                       cv::FileStorage::WRITE);
    fs << "total_events_per_pixel" << total_events_per_pixel;
    fs << "pos_events_per_pixel" << pos_events_per_pixel;
    fs << "neg_events_per_pixel" << neg_events_per_pixel;

    cv::Mat total_events_per_pixel_per_second;
    cv::Mat pos_events_per_pixel_per_second;
    cv::Mat neg_events_per_pixel_per_second;
    cv::multiply(cv::Mat::ones(rows, cols, CV_64F), total_events_per_pixel,
                 total_events_per_pixel_per_second, 1 / seconds);
    cv::multiply(cv::Mat::ones(rows, cols, CV_64F), pos_events_per_pixel,
                 pos_events_per_pixel_per_second, 1 / seconds);
    cv::multiply(cv::Mat::ones(rows, cols, CV_64F), neg_events_per_pixel,
                 neg_events_per_pixel_per_second, 1 / seconds);

    cv::minMaxIdx(total_events_per_pixel_per_second, &min, &max);
    std::cout << "Min: " << min << ", max: " << max << std::endl;
    cv::normalize(total_events_per_pixel_per_second, output, 0.0, 255.0,
                  cv::NORM_MINMAX, CV_8U);

    cv::imwrite(simulator_name + "events_per_pixel_per_second_from_sim.png",
                output);
    fs << "total_events_per_pixel_per_second"
       << total_events_per_pixel_per_second;
    fs << "pos_events_per_pixel_per_second" << pos_events_per_pixel_per_second;
    fs << "neg_events_per_pixel_per_second" << neg_events_per_pixel_per_second;
    fs.release();
  }
}

VideoStreamer::VideoStreamer(std::shared_ptr<EventSimulator> event_simulator,
                             std::shared_ptr<System> system)
    : event_simulator_{event_simulator}, system_{system}, roi_{cv::Rect(0, 0, 0, 0)} {}

void VideoStreamer::simulateFromStream(const int source_index)
{
  cv::VideoCapture cap(source_index);
  if (!cap.isOpened())
  {
    throw std::runtime_error("Error opening video source. Does it exist?");
  }

  cv::Mat frame, prev_frame, out_frame, dummyFlowFrame;
  float time_start, time_previous, time_current;
  cap.read(prev_frame);

  time_previous = 0.f;
  time_start = round4(static_cast<float>(std::chrono::duration_cast<std::chrono::duration<float>>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count()));
  prev_frame = toGray(prev_frame);

  event_simulator_->setup(prev_frame.size());

  while (true)
  {
    cap >> frame;
    time_current = round4(static_cast<float>(std::chrono::duration_cast<std::chrono::duration<float>>(
                                                 std::chrono::system_clock::now().time_since_epoch())
                                                 .count())) -
                   time_start;
    frame = toGray(frame);

    // int num_frames;
    std::vector<Event> &events = event_simulator_->getEvents(prev_frame, frame, time_previous, time_current,
                                                             roi_, true, out_frame, false, dummyFlowFrame); ////////////////////////////////

    // global alignment rotation estimate
    if (hasSystem())
    {
      system_->BindEvent(events);
    }

    bool should_quit = false;
    // for (int i = 0; i < num_frames; i++){
    cv::imshow("OpenCVPlayer", out_frame);
    char c = static_cast<char>(cv::waitKey(1));
    if (c == 27)
    { // esc. to quit
      should_quit = true;
    }
    //}

    if (should_quit)
    {
      break;
    }

    prev_frame = frame;
    time_previous = time_current;
  }
  cv::destroyAllWindows();
}

bool EventDataPlayer::readEventFile(const std::string &filePath)
{
  uint32_t reserve_size = static_cast<uint32_t>(2e8); // reserve size;

  // 從檔案讀回內容
  std::ifstream infile(filePath);
  if (!infile.is_open())
  {
    std::cout << "Error: Could not open event file for reading.\n";
    return false;
  }

  eventData.reserve(reserve_size);
  std::cout << "Reading data from file: " << filePath << "\n";
  std::cout << "eventData.capacity(): " << eventData.capacity() << "\n";

  size = 0;
  std::string line;
  while (getline(infile, line))
  {
    std::stringstream ss(line);
    float timestamp_;
    int x_, y_;
    bool polarity_;

    if (ss >> timestamp_ >> x_ >> y_ >> polarity_)
    {
      eventData.emplace_back(Event(x_, y_, timestamp_, polarity_));
      size++;
    }
    else
    {
      std::cerr << "資料格式錯誤： " << line << std::endl;
      return false;
    };
  }
  infile.close();
  std::cout << "Evet dataset read finish. Total " << size << " events.\n\n";

  return true;
}

int EventDataPlayer::readImagesTimestamp(const std::string &filePath)
{
  // 從檔案讀回內容
  std::ifstream infile(filePath);
  if (!infile.is_open())
  {
    std::cout << "Error: Could not open images.txt file for reading.\n";
    return false;
  }

  std::cout << "Reading image timestamp from file: " << filePath << "\n";

  size = 0;
  std::string line;
  while (getline(infile, line))
  {
    std::stringstream ss(line);
    float timestamp_;
    std::string image_path;

    if (ss >> timestamp_ >> image_path)
    {
      imageTimestamp.emplace_back(timestamp_);
      size++;
    }
    else
    {
      std::cerr << "資料格式錯誤： " << line << std::endl;
      return false;
    };
  }
  infile.close();
  std::cout << "images.txt read finish. Total " << size << " image timestamps.\n\n";

  return size;
}

float EventDataPlayer::getImageTimestamp(const int &index)
{
  if (index > imageTimestamp.size())
  {
    throw std::runtime_error("Invalid index, exceeds timestamp size.");
  }
  else
  {
    return imageTimestamp[index];
  }
}

cv::Mat EventDataPlayer::getSingleAccmulateFrame(const int &row, const int &col, const float &timestamp_prev, const float &timestamp_current)
{
  cv::Mat lighter_frame = cv::Mat::zeros(cv::Size(col, row), CV_8UC1);
  cv::Mat darker_frame = cv::Mat::zeros(cv::Size(col, row), CV_8UC1);
  cv::Mat zeros = cv::Mat::zeros(cv::Size(col, row), CV_8UC1);
  cv::Mat result;
  uchar *darker_frame_ptr = darker_frame.ptr<uchar>();
  uchar *lighter_frame_ptr = lighter_frame.ptr<uchar>();

  // 使用lambda function判斷時間戳
  auto lower = std::lower_bound(eventData.begin(), eventData.end(), timestamp_prev,
                                [](const Event &e, const float &ts)
                                { return e.timestamp < ts; });

  auto upper = std::upper_bound(eventData.begin(), eventData.end(), timestamp_current,
                                [](const float &ts, const Event &e)
                                { return ts < e.timestamp; });

  std::vector<Event> filtered_events(lower, upper); // 符合timestamp_prev到timestamp_current的事件向量

  for (const auto &iter : filtered_events)
  {
    if (roi_ == cv::Rect(0, 0, 0, 0))
    {
      // check bound
      if (iter.x > (col - 1) || iter.y > (row - 1) || iter.x < 0 || iter.y < 0)
      {
        continue;
      }

      // postive event or negative event!
      if (iter.polarity == true)
      {
        *(lighter_frame_ptr + iter.y * col + iter.x) = 255;
      }
      else
      {
        *(darker_frame_ptr + iter.y * col + iter.x) = 255;
      }
    }
    else
    {
      // check bound
      if (iter.x > (roi_.x + roi_.width - 1) || iter.y > (roi_.y + roi_.height - 1) || iter.x < roi_.x || iter.y < roi_.y)
      {
        continue;
      }

      // postive event or negative event!
      if (iter.polarity == true)
      {
        *(lighter_frame_ptr + iter.y * col + iter.x) = 255;
      }
      else
      {
        *(darker_frame_ptr + iter.y * col + iter.x) = 255;
      }
    }
  }

  std::vector<cv::Mat> channels;

  ////移除擷取roi邊緣，以免影響相機校正
  channels.push_back(lighter_frame);
  channels.push_back(zeros);
  channels.push_back(darker_frame);
  
  /*
  if (roi_ == cv::Rect(0, 0, 0, 0))
  {
    channels.push_back(lighter_frame);
    channels.push_back(zeros);
    channels.push_back(darker_frame);
  }
  else
  {
    channels.push_back(cv::Mat(lighter_frame, roi_));
    channels.push_back(cv::Mat(zeros, roi_));
    channels.push_back(cv::Mat(darker_frame, roi_));
  }
  */
  merge(channels, result);

  return result;
}

double EventDataPlayer::getInterval(const float &timestamp_prev, const float &timestamp_current)
{
  // 使用lambda function判斷時間戳
  auto lower = std::lower_bound(eventData.begin(), eventData.end(), timestamp_prev,
                                [](const Event &e, const float &ts)
                                { return e.timestamp < ts; });

  auto upper = std::upper_bound(eventData.begin(), eventData.end(), timestamp_current,
                                [](const float &ts, const Event &e)
                                { return ts < e.timestamp; });

  return static_cast<double>(std::distance(lower, upper)); // lower到upper之間的元素數量
}

void EventDataPlayer::printIndexEvent(const uint32_t &index)
{
  std::cout << "event[" << index << "]: (" << eventData[index].timestamp << ", "
            << eventData[index].x << ", "
            << eventData[index].y << ", "
            << eventData[index].polarity << ")\n";
}

void EventDataPlayer::publishData(const uint16_t &bunchSize)
{
  uint32_t iter = 0;
  uint32_t iter_size = size / bunchSize;

  std::cout << "Publish event data with bunch size " << bunchSize << " per iteration...\n";

  uint32_t start = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

  // 輸入事件到system進行運算
  for (uint32_t i = 0; i < iter_size; i++)
  {
    std::vector<Event> temp(eventData.begin() + iter, eventData.begin() + iter + bunchSize);
    system_->BindEvent(temp);
    iter += bunchSize;
  }
  if (iter < size) // 補上iter_size餘數
  {
    std::vector<Event> temp(eventData.begin() + iter, eventData.end());
    system_->BindEvent(temp);
  }

  uint32_t end = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  uint32_t ms = (end - start);

  std::cout << "Dataset simulate in " << static_cast<double>(ms) / 1000. << " seconds.\n";
  std::cout << "Dataset simulate finish.\n";
}