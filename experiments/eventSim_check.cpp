#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "EventSimulators.h"

/**
 * 檢查事件模擬器的插值方向錯誤
 */

int main(int argc, char **argv)
{

    std::string video_path("../../experiments/testframe/testframe_video.mp4");
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cout << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "Video size: " << width << "x" << height << ", frames: " << frame_count << std::endl;

    cv::Mat prev_frame, frame, frameEvent, frameFlow;
    std::vector<Event> events;

    // std::shared_ptr<SparseOpticalFlowCalculator> ofc_lk = std::make_shared<LKOpticalFlowCalculator>();
    // std::shared_ptr<EventSimulator> ssmes = std::make_shared<SparseInterpolatedEventSimulator>(ofc_lk, 0, 10, 10);

     std::shared_ptr<DenseOpticalFlowCalculator> ofc_fb = std::make_shared<FarnebackFlowCalculator>();
     std::shared_ptr<EventSimulator> dsmes = std::make_shared<DenseInterpolatedEventSimulator>(ofc_fb, 2, 10, 10);

    // dis跑小圖要記得去設定DISOpticalFlow的參數 DISOpticalFlow_->setPatchSize(4);
    //std::shared_ptr<DenseOpticalFlowCalculator> ofc_dis_med = std::make_shared<DISOpticalFlowCalculator>(cv::DISOpticalFlow::PRESET_MEDIUM);
    //std::shared_ptr<EventSimulator> dsmes = std::make_shared<DenseInterpolatedEventSimulator>(ofc_dis_med, 2, 10, 10);

    // 讀第一張
    cap >> prev_frame;
    if (prev_frame.empty())
    {
        std::cout << "No frames in video.\n";
        return -1;
    }
    cv::cvtColor(prev_frame, prev_frame, cv::COLOR_BGR2GRAY);

    int idx = 1;
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::namedWindow("Event Frame", cv::WINDOW_NORMAL);
    cv::namedWindow("Flow Frame", cv::WINDOW_NORMAL);
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // 產生事件
        events.clear();
        events = dsmes->getEvents(prev_frame, frame, idx - 1, idx, cv::Rect(0, 0, 0, 0), true, frameEvent, true, frameFlow);

        // 輸出事件
        std::cout << "Frame " << idx << " events:\n";
        for (const auto &e : events)
        {
            std::cout << "x=" << e.x << ", y=" << e.y << ", t=" << e.timestamp << ", p=" << e.polarity << "\n";
        }

        cv::imshow("Video Frame", prev_frame);
        cv::imshow("Event Frame", frameEvent);
        cv::imshow("Flow Frame", frameFlow);
        //cv::waitKey(0);
        prev_frame = frame;
        idx++;
    }

    return 0;
}