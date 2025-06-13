#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>

#include "EventSimulators.h"
#include "Player.h"

/**
 *輸出ground truth與sim的Event Per Pixel Per Second
 */

const int c_pos_param[] = {5};
const int c_neg_param[] = {5};
const int num_inter_param[] = {8};

int main(int argc, char **argv)
{
    cv::Mat frame_prev, frame_curr, eventFrame_sim, eventFrame_gt, flowFrame;

    int size_timestamp = 0;
    int idx_timestamp = 0;
    int time_second = 0; // 用於計算EPS
    float frameTime_prev = 0.f;
    float frameTime_curr = 0.f;
    double rmse_sum, rmse_avg, eps;
    std::vector<double> eps_gt, eps_sim;

    std::string event_path = "../../experiments/events/events_shapes.txt";
    // std::string event_path = "../../experiments/events/events_boxes.txt";
    //  std::string event_path = "../../experiments/events/events_poster.txt";

    std::string imageTimestamp_path = "../../experiments/images/images_shapes.txt";
    // std::string imageTimestamp_path = "../../experiments/images/images_boxes.txt";
    // std::string imageTimestamp_path = "../../experiments/images/images_poster.txt";

    std::string video_path = "../../experiments/videos/shapes_rotation_video.mp4";
    // std::string video_path = "../../experiments/videos/boxes_rotation_video.mp4";
    //  std::string video_path = "../../experiments/videos/poster_rotation_video.mp4";

    std::string gt_path("./images/groundTruth");

    //// 1.產生ground truth event frame
    // 讀取ground truth事件檔
    EventDataPlayer eventPlayer(cv::Rect(0, 0, 0, 0)); // 初始化不使用roi

    auto time_prev = std::chrono::high_resolution_clock::now();
    if (eventPlayer.readEventFile(event_path))
    {
        std::cout << "Successful read event file!\n";
    }
    else
    {
        std::cout << "Can't open event file. Check evnet file path!\n";
        return -1;
    };

    size_timestamp = eventPlayer.readImagesTimestamp(imageTimestamp_path);
    if (size_timestamp > 0)
    {
        std::cout << "Successful read images.txt for timestamp!\n";
    }
    else
    {
        std::cout << "Can't open images.txt. Check images.txt path!\n";
        return -1;
    };

    auto time_curr = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(time_curr - time_prev);
    std::cout << "Spend " << static_cast<double>(duration.count()) << " seconds for open event file!\n\n";

    // 讀取影片檔
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cout << "can't open video, check video path.\n";
        return -1;
    }
    int size_cap = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cv::namedWindow("gt");
    cv::namedWindow("sim");

    cap >> frame_curr;
    frameTime_prev = eventPlayer.getImageTimestamp(idx_timestamp);
    // std::cout<<"first cap timestamp: "<<frameTime_prev<<"\n";

    eps = 0.;
    for (;;)
    {
        cap >> frame_curr;
        idx_timestamp += 1;
        frameTime_curr = eventPlayer.getImageTimestamp(idx_timestamp);
        // std::cout<<"time interval: "<<frameTime_curr - frameTime_prev<<"ms\n";
        // std::cout<<"current timestamp: "<<frameTime_curr<<"\n";

        if (frame_curr.empty())
            break;

        // ground truth事件影像
        eventFrame_gt = eventPlayer.getSingleAccmulateFrame(height, width, frameTime_prev, frameTime_curr);

        // 紀錄EPS
        if (frameTime_curr >= static_cast<float>(time_second + 1))
        {
            std::cout << "frameTime_curr: " << frameTime_curr << ", eps: " << eps << "\n";

            time_second++;
            eps_gt.push_back(eps);
            eps = 0.;
        }
        eps += static_cast<double>(eventPlayer.getInterval(frameTime_prev, frameTime_curr)); // EPS from ground truth event data

        // 顯示ground truth事件影像
        // imshow("cap", frame_curr);
        // imshow("gt", eventFrame_gt);
        // cv::waitKey(1);

        frameTime_prev = frameTime_curr;
    }

    //// 2.儲存gt EPS資料
    std::filesystem::remove_all(gt_path);
    std::filesystem::create_directories(gt_path);

    std::ofstream eps_file(std::string("gtEPS.txt"));
    for (const auto &val : eps_gt)
    {
        eps_file << val << "\n";
    }
    eps_file.close();

    std::cout << "video size:" << size_cap << "\n";
    // std::cout << "timestamp size:" << size_timestamp << "\n";

    //// 3.sim EPS
    time_second = 0; // 重置時間秒數
    eps = 0.;

    int size_cpos = sizeof(c_pos_param) / sizeof(c_pos_param[0]);
    int size_cneg = sizeof(c_neg_param) / sizeof(c_neg_param[0]);
    int size_num_inter = sizeof(num_inter_param) / sizeof(num_inter_param[0]);

    // 設定基本參數
    std::shared_ptr<DenseOpticalFlowCalculator> ofc_dis = std::make_shared<DISOpticalFlowCalculator>(cv::DISOpticalFlow::PRESET_MEDIUM);

    for (int i_cpos = 0; i_cpos < size_cpos; i_cpos++)
    {
        for (int i_cneg = 0; i_cneg < size_cneg; i_cneg++)
        {
            for (int i_num_inter = 0; i_num_inter < size_num_inter; i_num_inter++)
            {
                // 建立事件模擬器
                std::shared_ptr<EventSimulator> disESim = std::make_shared<DenseInterpolatedEventSimulator>(ofc_dis, num_inter_param[i_num_inter], c_pos_param[i_cpos], c_neg_param[i_cneg]);

                //// 5.產生模擬事件影像
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 影片串流回到第0張
                idx_timestamp = 0;                   // 重置時間戳

                cap >> frame_prev;
                frameTime_prev = eventPlayer.getImageTimestamp(idx_timestamp);
                // std::cout << "first capture timestamp: " << frameTime_prev << "\n";

                cv::cvtColor(frame_prev, frame_prev, cv::COLOR_BGR2GRAY);

                for (int i = 0; i < size_cap - 1; i++)
                {
                    cap >> frame_curr;
                    idx_timestamp += 1;
                    frameTime_curr = eventPlayer.getImageTimestamp(idx_timestamp);
                    // std::cout<<"current timestamp: "<<frameTime_curr<<"\n";
                    // std::cout<<"time interval: "<<frameTime_curr - frameTime_prev<<"ms\n";

                    cv::cvtColor(frame_curr, frame_curr, cv::COLOR_BGR2GRAY);

                    // 取出ground truth事件影像
                    eventFrame_gt = eventPlayer.getSingleAccmulateFrame(height, width, frameTime_prev, frameTime_curr);

                    // 計算模擬事件、累積事件影像、光流
                    std::vector<Event> &event_sim = disESim->getEvents(frame_prev, frame_curr, frameTime_prev, frameTime_curr,
                                                                       cv::Rect(0, 0, 0, 0), true, eventFrame_sim, false, flowFrame); ///// 模擬器的roi在這裡

                    //// EPS
                    if (frameTime_curr >= static_cast<float>(time_second + 1))
                    {
                        std::cout << "frameTime_curr: " << frameTime_curr << ", eps: " << eps << "\n";

                        time_second++;
                        eps_sim.push_back(eps);
                        eps = 0;
                    }
                    eps += static_cast<double>(event_sim.size());

                    cv::imshow("gt", eventFrame_gt);
                    cv::imshow("sim", eventFrame_sim);
                    // cv::imshow("flow", flowFrame);
                    cv::waitKey(1);

                    frame_prev = frame_curr;
                    frameTime_prev = frameTime_curr;
                }

                //// 7.儲存sim EPS資料
                std::stringstream ss;
                ss << "simEPS_pos" << c_pos_param[i_cpos]
                   << "_neg" << c_neg_param[i_cneg] << "_inter" << num_inter_param[i_num_inter] << ".txt";

                std::string eps_txt_path = ss.str();
                std::ofstream eps_file(eps_txt_path);
                for (const auto &val : eps_sim)
                {
                    eps_file << val << "\n";
                }
                eps_file.close();
            }
        }
    }

    return 0;
}
