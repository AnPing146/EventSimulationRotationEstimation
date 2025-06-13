#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>

#include "EventSimulators.h"
#include "Player.h"

/** Event simulator parameters experiments
 *多組實驗:
 * 1.由不同參數組產生不同的模擬事件
 * 比較每張累積事件影像的RMSE，判斷事件產生位置正確性，同時檢查由錯誤光流產生的邊緣
 * 比較每張影像的EPF(event per pixel per frame)，確認總事件數量是否接近
 *
 *改進實驗:
 * 1.對較佳的RMSE組合，查看原始圖片，設定適當的roi
 * 2.更換資料組測試!
 *
 * 注意:因為影片檔的時間戳與ground truth的時間戳誤差太大，故不使用影片的時間戳，以ground truth為基準!
 * roi不應該變更影像尺寸，會影響到影像校正。計算RMSE時要取ROI，避免邊框降低數值。
 */


const int roi_bound_param[] = {0}; // roi_bound為百分比, 10表示截去邊緣佔影像百分之10
const int c_pos_param[] = {5, 10, 15};
const int c_neg_param[] = {5, 10, 15};
const int num_inter_param[] = {8, 10};

int main(int argc, char **argv)
{
    cv::Mat frame_prev, frame_curr, eventFrame_sim, eventFrame_gt, flowFrame;

    int size_timestamp = 0;
    int idx_timestamp = 0;
    float frameTime_prev = 0.f;
    float frameTime_curr = 0.f;
    double rmse_sum, rmse_avg, epf, epf_sum, epf_avg;
    std::vector<cv::Mat> eventFrame_gt_vec;
    std::vector<double> epf_vec;

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
    std::cout << "Spend " << static_cast<double>(duration.count()) << " seconds for open event file!\n";

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
    //cv::namedWindow("flow");

    cap >> frame_curr;
    frameTime_prev = eventPlayer.getImageTimestamp(idx_timestamp);
    // std::cout<<"first cap timestamp: "<<frameTime_prev<<"\n";

    epf = 0.;
    epf_sum = 0.;
    epf_avg = 0.;
    for (;;)
    {
        cap >> frame_curr;
        idx_timestamp += 1;
        frameTime_curr = eventPlayer.getImageTimestamp(idx_timestamp);
        // std::cout<<"time interval: "<<frameTime_curr - frameTime_prev<<"ms\n";
        // std::cout<<"current timestamp: "<<frameTime_curr<<"\n";

        if (frame_curr.empty())
            break;

        // ground truth事件轉影像
        eventFrame_gt = eventPlayer.getSingleAccmulateFrame(height, width, frameTime_prev, frameTime_curr);
        eventFrame_gt_vec.push_back(eventFrame_gt);
        epf = eventPlayer.getInterval(frameTime_prev, frameTime_curr) / static_cast<double>(width * height); // EPF = events / (pixel * sigleFrame)
        epf_vec.push_back(epf);

        // 顯示ground truth事件影像
        // imshow("cap", frame_curr);
        // imshow("gt", eventFrame_gt);
        // cv::waitKey(1);

        frameTime_prev = frameTime_curr;
        epf_sum += epf;
    }

    //// 2.儲存ground truth資料(影像與EPF)
    std::filesystem::remove_all(gt_path);
    std::filesystem::create_directories(gt_path);

    // 儲存gt影像
    for (int i = 0; i < eventFrame_gt_vec.size(); i++)
    {
        std::stringstream ss;
        ss << gt_path << "/frame_" << std::to_string(i) << ".jpg";
        cv::imwrite(ss.str(), eventFrame_gt_vec[i]);
    }
    // 儲存原始大小的gt EPF
    std::ofstream epf_file(std::string("gtEPF_roi0.txt"));
    for (const auto &val : epf_vec)
    {
        epf_file << val << "\n";
    }
    epf_file.close();
    std::cout << "gt epf_avg: " << epf_sum / static_cast<double>(eventFrame_gt_vec.size()) << "\n"; // AEPF = sigma(EPF) / frames

    std::cout << "video size:" << size_cap << "\n";
    std::cout << "gt frame size:" << eventFrame_gt_vec.size() << "\n\n";
    // std::cout << "timestamp size:" << size_timestamp << "\n";

    //// 3.多組實驗
    int size_cpos = sizeof(c_pos_param) / sizeof(c_pos_param[0]);
    int size_cneg = sizeof(c_neg_param) / sizeof(c_neg_param[0]);
    int size_num_inter = sizeof(num_inter_param) / sizeof(num_inter_param[0]);
    int size_roi_bound = sizeof(roi_bound_param) / sizeof(roi_bound_param[0]);

    // 設定基本參數
    std::shared_ptr<DenseOpticalFlowCalculator> ofc_dis = std::make_shared<DISOpticalFlowCalculator>(cv::DISOpticalFlow::PRESET_MEDIUM);

    for (int i_cpos = 0; i_cpos < size_cpos; i_cpos++)
    {
        for (int i_cneg = 0; i_cneg < size_cneg; i_cneg++)
        {
            for (int i_num_inter = 0; i_num_inter < size_num_inter; i_num_inter++)
            {
                for (int i_roi_bound = 0; i_roi_bound < size_roi_bound; i_roi_bound++)
                {
                    rmse_sum = 0.;
                    rmse_avg = 0.;
                    epf_sum = 0.;
                    epf_avg = 0.;
                    cv::Rect roi;
                    epf_vec.clear();

                    //// 4.重新設定ground truth roi
                    int x0 = static_cast<int>(width * (roi_bound_param[i_roi_bound] / 100.) / 2.);
                    int x1 = width - 2 * x0;
                    int y0 = static_cast<int>(height * (roi_bound_param[i_roi_bound] / 100.) / 2.);
                    int y1 = height - 2 * y0;
                    if (x0 == 0 && y0 == 0)
                    {
                        eventPlayer.setRoi(cv::Rect(0, 0, 0, 0));
                    }
                    else
                    {
                        roi = cv::Rect(x0, y0, x1, y1);
                        eventPlayer.setRoi(roi);
                    }
                    std::cout << "roi: [" << roi.x << ", " <<roi.y <<", " << roi.width<<", "<<roi.height << "]\n";

                    // 建立要存放模擬影像的資料夾
                    std::stringstream ss, ss_fileName;
                    ss << "./images/sim_roi" << roi_bound_param[i_roi_bound] << "_pos" << c_pos_param[i_cpos]
                       << "_neg" << c_neg_param[i_cneg] << "_inter" << num_inter_param[i_num_inter];
                    std::cout << ss.str() << "\n";
                    std::filesystem::remove_all(ss.str());
                    std::filesystem::create_directories(ss.str());

                    // 建立事件模擬器
                    std::shared_ptr<EventSimulator> disESim = std::make_shared<DenseInterpolatedEventSimulator>(ofc_dis, num_inter_param[i_num_inter], c_pos_param[i_cpos], c_neg_param[i_cneg]);

                    //// 5.產生模擬事件影像
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 影片串流回到第0張
                    idx_timestamp = 0;                   // 重置時間戳

                    cap >> frame_prev;
                    frameTime_prev = eventPlayer.getImageTimestamp(idx_timestamp);
                    //std::cout << "first capture timestamp: " << frameTime_prev << "\n";

                    cv::cvtColor(frame_prev, frame_prev, cv::COLOR_BGR2GRAY);

                    for (int i = 0; i < size_cap - 1; i++)
                    {
                        cap >> frame_curr;
                        idx_timestamp += 1;
                        frameTime_curr = eventPlayer.getImageTimestamp(idx_timestamp);
                        // std::cout<<"current timestamp: "<<frameTime_curr<<"\n";
                        // std::cout<<"time interval: "<<frameTime_curr - frameTime_prev<<"ms\n";

                        cv::cvtColor(frame_curr, frame_curr, cv::COLOR_BGR2GRAY);

                        // 取出ground truth影像
                        eventFrame_gt = eventPlayer.getSingleAccmulateFrame(height, width, frameTime_prev, frameTime_curr);

                        // 計算模擬事件、累積事件影像、光流
                        std::vector<Event> &event_sim = disESim->getEvents(frame_prev, frame_curr, frameTime_prev, frameTime_curr,
                                                                           roi, true, eventFrame_sim, true, flowFrame); ///// 模擬器的roi在這裡

                        //// 6.評估指標 (比較每張影像RMSE、EPF)
                        // RMSE
                        cv::Mat diff, diff_squared;
                        if (x0 == 0 && y0 == 0)
                        {
                            cv::absdiff(eventFrame_sim, eventFrame_gt, diff);                               // |I1 - I2|
                        }
                        else
                        {
                            cv::absdiff(eventFrame_sim(roi), eventFrame_gt(roi), diff);                     // |I1 - I2|
                        }
                        diff.convertTo(diff, CV_32F);                                                       // 防止 overflow
                        cv::multiply(diff, diff, diff_squared);                                             // (I1 - I2)^2

                        cv::Scalar mse = cv::mean(diff_squared);
                        rmse_sum += std::sqrt((mse[0] + mse[1] + mse[2]) / 3.0);

                        // EPF
                        if (x0 == 0 && y0 == 0)
                        {
                            epf = static_cast<double>(event_sim.size()) / static_cast<double>(width * height); // EPF = events / (pixel * sigleFrame)
                        }
                        else
                        {
                            epf = static_cast<double>(event_sim.size()) / static_cast<double>(x1 * y1);
                        }
                        epf_sum += epf;
                        epf_vec.push_back(epf);

                        // 儲存影像
                        ss_fileName.str("");
                        ss_fileName.clear();
                        ss_fileName << ss.str() << "/frame_" << std::to_string(i) << ".jpg";
                        //std::cout << ss_fileName.str() << "\n";

                        cv::imwrite(ss_fileName.str(), eventFrame_sim);

                        cv::imshow("gt", eventFrame_gt);
                        cv::imshow("sim", eventFrame_sim);
                        //cv::imshow("flow", flowFrame);
                        cv::waitKey(1);

                        frame_prev = frame_curr;
                        frameTime_prev = frameTime_curr;
                    }

                    rmse_avg = rmse_sum / static_cast<double>(size_cap - 1);
                    epf_avg =  epf_sum / static_cast<double>(size_cap - 1);
                    std::cout << "rmse_avg:" << rmse_avg << ", epf_avg: " << epf_avg << "\n\n";

                    //// 7.儲存epf_vec
                    ss.str("");
                    ss.clear();
                    ss << "simEPF_roi" << roi_bound_param[i_roi_bound] << "_pos" << c_pos_param[i_cpos]
                       << "_neg" << c_neg_param[i_cneg] << "_inter" << num_inter_param[i_num_inter] << ".txt";

                    std::string epf_txt_path = ss.str();
                    std::ofstream epf_file(epf_txt_path);
                    for (const auto &val : epf_vec)
                    {
                        epf_file << val << "\n";
                    }
                    epf_file.close();
                }
            }
        }
    }

    return 0;
}

/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "EventSimulators.h"

int main(int argc, char **argv)
{

    std::string video_path("../../experiments/videos/testframe_video.mp4");
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

    cv::Mat prev_frame, frame, frameEvent;
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
    cv::namedWindow("Event Frame", cv::WINDOW_NORMAL);
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // 產生事件
        events.clear();
        events = dsmes->getEvents(prev_frame, frame, idx - 1, idx, cv::Rect(0, 0, 0, 0), true, frameEvent, false, frame);

        // 輸出事件
        std::cout << "Frame " << idx << " events:\n";
        for (const auto &e : events)
        {
            std::cout << "x=" << e.x << ", y=" << e.y << ", t=" << e.timestamp << ", p=" << e.polarity << "\n";
        }

        cv::imshow("Event Frame", frameEvent);
        cv::waitKey(0);
        prev_frame = frame.clone();
        idx++;
    }

    return 0;
}
*/