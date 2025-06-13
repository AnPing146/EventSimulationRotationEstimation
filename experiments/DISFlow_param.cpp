#include "flowColor.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

/** DIS optical flow parameter experiments
 * -----可調參數-----
 * theta_sf: 金字塔最細使用到哪一層。 數值越小越細緻，但0會出現方格化，需增加theta_it來補救
 * theta_ps: patch大小。 原作者建議8~12，因此固定8  //不調整!!
 * theta_ov: patch重疊pixel。 重疊越大越細緻
 * theta_it: 每patch的梯度下降迭代次數。 越大越細緻，但越慢
 * theta_vo, theta_vi: 變分法細化迭代次數。 越大越細緻，但越慢
 * theta_delta: 變分法細化的強度權重。             //不調整!!
 * theta_gamma: 變分法細化的梯度權重。
 * theta_alpha: 變分法細化的平滑度權重。
 *
 * -----自動參數-----
 * theta_sd: 金字塔下取樣分母。 預設為2
 * theta_ss: 金字塔層數。 公式:"theta_ss = log(2 * width / (f * theta_ps)) / log(theta_sd)",
 * 其中f為希望估測的最大運動佔影像寬度的比例(例如:10表示10%)
 *
 * -----實驗設定-----
 * 第一次(主參數)
 * 第幾組        1  |  2  |  3  | 4
 * theta_sf     0     0     1    1
 * theta_ov     4     2     4    2
 * theta_it     16    64    64   16
 *
 * 第二次(變分法參數)
 * 第幾組        1  |  2  |  3  | 4
 * theta_vio     0    0     5    5
 * theta_gamma   10   40    10   40
 * theta_alpha   40   10    10   40
 */

const int param1[4][3] = {{0, 4, 16}, {0, 2, 64}, {1, 4, 64}, {1, 2, 16}};
const int param2[4][3] = {{0, 10, 40}, {0, 40, 10}, {5, 10, 10}, {5, 40, 40}};
const bool multipleExp = true; // true 多次實驗或單次實驗. false 單次實驗用於最後驗證(同時輸出色環)
const int i_param1 = 0;         // 設定單次實驗分別要用哪組參數 (記得從0開始)
const int i_param2 = 3;

int main(int argc, char **argv)
{
    std::string image_floder_path("../../experiments/images/alley_1");
    std::string gt_path("../../experiments/images/alley_1/flo");
    const uint saveFrameIndex = 22; // 要儲存的影像索引

    cv::Mat img, gray, img_prev, gray_prev;
    cv::Mat flow_adjust, flow_gt, frame_adjust, frame_gt;
    double mepe;

    //// 讀取ground-truth與影像串列
    std::vector<cv::Mat> flow_gt_vec = readFrame(gt_path, ".flo");
    std::cout << "flow_gt.size(): " << flow_gt_vec.size() << ", ";
    std::vector<cv::Mat> frame_vec = readFrame(image_floder_path, ".png");
    std::cout << "frames.size(): " << frame_vec.size() << "\n\n";

    ////參數實驗: MEPE Time imwrite
    if (multipleExp == true)
    {
        int rows;
        for (int i_exp = 0; i_exp < 2; i_exp++)
        {
            cv::Ptr<cv::DISOpticalFlow> DIS_adjust = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

            if (i_exp == 0) // 第一次主參數
            {
                rows = sizeof(param1) / sizeof(param1[0]);
                for (int i_param = 0; i_param < rows; i_param++)
                {
                    DIS_adjust->setFinestScale(param1[i_param][0]);
                    DIS_adjust->setPatchStride(param1[i_param][1]);
                    DIS_adjust->setGradientDescentIterations(param1[i_param][2]);
                    std::cout << "[Param] sf: " << DIS_adjust->getFinestScale()
                              << ", ov: " << DIS_adjust->getPatchStride()
                              << ", it: " << DIS_adjust->getGradientDescentIterations() << "\n";

                    //// 1.運算MEPE
                    mepe = 0;
                    for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
                    {
                        if (i == 1)
                        {
                            img_prev = frame_vec[0];
                        }
                        img = frame_vec[i];
                        cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                        DIS_adjust->calc(gray_prev, gray, flow_adjust);
                        frame_adjust = flowToColor(flow_adjust);
                        flow_gt = flow_gt_vec[i - 1]; // 0~48

                        mepe += meanEndPointError(flow_adjust, flow_gt);

                        cv::imshow("frame", frame_adjust);
                        cv::waitKey(1);

                        img_prev = img;
                    }
                    mepe /= frame_vec.size() - 1;
                    std::cout << "MEPE: " << mepe << "\n";

                    //// 2.運算Time
                    std::chrono::milliseconds total_duration(0);
                    for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
                    {
                        if (i == 1)
                        {
                            img_prev = frame_vec[0];
                        }
                        img = frame_vec[i];
                        cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                        auto t1 = std::chrono::high_resolution_clock::now(); // 單次計時
                        DIS_adjust->calc(gray_prev, gray, flow_adjust);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                        total_duration += duration; // 累計時間

                        frame_adjust = flowToColor(flow_adjust);
                        flow_gt = flow_gt_vec[i - 1]; // 0~48

                        mepe += meanEndPointError(flow_adjust, flow_gt);

                        // cv::imshow("frame", frame_adjust);
                        // cv::waitKey(1);

                        img_prev = img;
                    }
                    double time_average = static_cast<double>(total_duration.count() / (frame_vec.size() - 1));
                    std::cout << "Average DIS-flow process time per frame: " << time_average << "ms\n\n";

                    //// 3.save single frame
                    std::stringstream ss;
                    ss << "DIS_idx" << saveFrameIndex << "_exp" << i_param + 1
                       << "_sf" << DIS_adjust->getFinestScale()
                       << "_ov" << DIS_adjust->getPatchStride()
                       << "_it" << DIS_adjust->getGradientDescentIterations() << ".jpg";
                    std::string fileName = ss.str();

                    img_prev = frame_vec[saveFrameIndex];
                    img = frame_vec[saveFrameIndex + 1];
                    cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    DIS_adjust->calc(gray_prev, gray, flow_adjust);
                    frame_adjust = flowToColor(flow_adjust);
                    frame_adjust.convertTo(frame_adjust, CV_8UC3, 255.0);

                    cv::imwrite(fileName, frame_adjust);
                }
            }
            else if (i_exp == 1) // 第二次變分法參數
            {
                rows = sizeof(param2) / sizeof(param2[0]);
                for (int i_param = 0; i_param < rows; i_param++)
                {
                    DIS_adjust->setVariationalRefinementIterations(param2[i_param][0]);
                    DIS_adjust->setVariationalRefinementGamma(param2[i_param][1]);
                    DIS_adjust->setVariationalRefinementAlpha(param2[i_param][2]);
                    std::cout << "[Param] vio: " << DIS_adjust->getVariationalRefinementIterations()
                              << ", gamma: " << DIS_adjust->getVariationalRefinementGamma()
                              << ", alpha: " << DIS_adjust->getVariationalRefinementAlpha() << "\n";

                    //// 1.運算MEPE
                    mepe = 0;
                    for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
                    {
                        if (i == 1)
                        {
                            img_prev = frame_vec[0];
                        }
                        img = frame_vec[i];
                        cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                        DIS_adjust->calc(gray_prev, gray, flow_adjust);
                        frame_adjust = flowToColor(flow_adjust);
                        flow_gt = flow_gt_vec[i - 1]; // 0~48

                        mepe += meanEndPointError(flow_adjust, flow_gt);

                        cv::imshow("frame", frame_adjust);
                        cv::waitKey(1);

                        img_prev = img;
                    }
                    mepe /= frame_vec.size() - 1;
                    std::cout << "MEPE: " << mepe << "\n";

                    //// 2.運算Time
                    std::chrono::milliseconds total_duration(0);
                    for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
                    {
                        if (i == 1)
                        {
                            img_prev = frame_vec[0];
                        }
                        img = frame_vec[i];
                        cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                        auto t1 = std::chrono::high_resolution_clock::now(); // 單次計時
                        DIS_adjust->calc(gray_prev, gray, flow_adjust);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                        total_duration += duration; // 累計時間

                        frame_adjust = flowToColor(flow_adjust);
                        flow_gt = flow_gt_vec[i - 1]; // 0~48

                        mepe += meanEndPointError(flow_adjust, flow_gt);

                        img_prev = img;
                    }
                    double time_average = static_cast<double>(total_duration.count() / (frame_vec.size() - 1));
                    std::cout << "Average DIS-flow process time per frame: " << time_average << "ms\n\n";

                    //// 3.save single frame
                    std::stringstream ss;
                    ss << "DIS_index" << saveFrameIndex << "_exp" << i_param + 1
                       << "_vio" << DIS_adjust->getVariationalRefinementIterations()
                       << "_gamma" << DIS_adjust->getVariationalRefinementGamma()
                       << "_alpha" << DIS_adjust->getVariationalRefinementAlpha() << ".jpg";
                    std::string fileName = ss.str();

                    img_prev = frame_vec[saveFrameIndex];
                    img = frame_vec[saveFrameIndex + 1];
                    cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    DIS_adjust->calc(gray_prev, gray, flow_adjust);
                    frame_adjust = flowToColor(flow_adjust);
                    frame_adjust.convertTo(frame_adjust, CV_8UC3, 255.0);

                    cv::imwrite(fileName, frame_adjust);

                    //// 4.save ground truth frame
                    ss.str("");
                    ss.clear();
                    ss << "DIS_index" << saveFrameIndex << "_gt.jpg";
                    fileName = ss.str();

                    flow_gt = flow_gt_vec[saveFrameIndex];
                    frame_gt = flowToColor(flow_gt);
                    frame_gt.convertTo(frame_gt, CV_8UC3, 255.0);

                    cv::imwrite(fileName, frame_gt);
                }
            }

            DIS_adjust.release(); // 收回DIS物件，並於新迴圈重新初始化
        }
    }
    else //// 單次實驗用於最後驗證
    {
        cv::Ptr<cv::DISOpticalFlow> DIS_adjust = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

        DIS_adjust->setFinestScale(param1[i_param1][0]);
        DIS_adjust->setPatchStride(param1[i_param1][1]);
        DIS_adjust->setGradientDescentIterations(param1[i_param1][2]);
        std::cout << "[Param] sf: " << DIS_adjust->getFinestScale()
                  << ", ov: " << DIS_adjust->getPatchStride()
                  << ", it: " << DIS_adjust->getGradientDescentIterations() << "\n";
        DIS_adjust->setVariationalRefinementIterations(param2[i_param2][0]);
        DIS_adjust->setVariationalRefinementGamma(10); // param2[i_param2][1]
        DIS_adjust->setVariationalRefinementAlpha(param2[i_param2][2]);
        std::cout << "[Param] vio: " << DIS_adjust->getVariationalRefinementIterations()
                  << ", gamma: " << DIS_adjust->getVariationalRefinementGamma()
                  << ", alpha: " << DIS_adjust->getVariationalRefinementAlpha() << "\n";

        //// 1.運算MEPE
        mepe = 0;
        for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
        {
            if (i == 1)
            {
                img_prev = frame_vec[0];
            }
            img = frame_vec[i];
            cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

            DIS_adjust->calc(gray_prev, gray, flow_adjust);
            frame_adjust = flowToColor(flow_adjust);
            flow_gt = flow_gt_vec[i - 1]; // 0~48

            mepe += meanEndPointError(flow_adjust, flow_gt);

            cv::imshow("frame", frame_adjust);
            cv::waitKey(1);

            img_prev = img;
        }
        mepe /= frame_vec.size() - 1;
        std::cout << "MEPE: " << mepe << "\n";

        //// 2.運算Time
        std::chrono::milliseconds total_duration(0);
        for (size_t i = 1; i < frame_vec.size(); ++i) // 1~49
        {
            if (i == 1)
            {
                img_prev = frame_vec[0];
            }
            img = frame_vec[i];
            cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

            auto t1 = std::chrono::high_resolution_clock::now(); // 單次計時
            DIS_adjust->calc(gray_prev, gray, flow_adjust);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            total_duration += duration; // 累計時間

            frame_adjust = flowToColor(flow_adjust);
            flow_gt = flow_gt_vec[i - 1]; // 0~48

            mepe += meanEndPointError(flow_adjust, flow_gt);

            // cv::imshow("frame", frame_adjust);
            // cv::waitKey(1);

            img_prev = img;
        }
        double time_average = static_cast<double>(total_duration.count() / (frame_vec.size() - 1));
        std::cout << "Average DIS-flow process time per frame: " << time_average << "ms\n\n";

        std::cout << "DISOpticalFlow \"MEDIUM\" Parameters:" << "\n";
        std::cout << "Finest Scale(theta_sf): " << DIS_adjust->getFinestScale() << "\n"; // PRESET_MEDIUM:1
        std::cout << "Patch Size(theta_ps): " << DIS_adjust->getPatchSize() << "\n";
        std::cout << "Patch Stride(theta_ov): " << DIS_adjust->getPatchStride() << "\n";                                                    // PRESET_MEDIUM:3
        std::cout << "Gradient Descent Iterations(theta_it): " << DIS_adjust->getGradientDescentIterations() << "\n";                       // PRESET_MEDIUM:25
        std::cout << "Variational Refinement Iterations(theta_vo, theta_vi): " << DIS_adjust->getVariationalRefinementIterations() << "\n"; // PRESET_MEDIUM:5
        std::cout << "Variational Refinement Delta(theta_delta): " << DIS_adjust->getVariationalRefinementDelta() << "\n";
        std::cout << "Variational Refinement Gamma(theta_gamma): " << DIS_adjust->getVariationalRefinementGamma() << "\n";   // PRESET_MEDIUM:10
        std::cout << "Variational Refinement Alpha(theta_alpha): " << DIS_adjust->getVariationalRefinementAlpha() << "\n\n"; // PRESET_MEDIUM:10

        DIS_adjust.release();

        cv::Mat colorMap = colorTest();
        colorMap.convertTo(colorMap, CV_8UC3, 255.0);
        cv::imwrite("colorMap.jpg", colorMap);
    }

    return 0;
}

/*
    cv::Mat img1, gray1;
    cv::Mat flow_basic, frame_adjust;

    cv::Ptr<cv::DISOpticalFlow> DIS_basic = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    // 讀取並輸出 DISOpticalFlow "PRESET_MEDIUM" 所有參數
    std::cout << "DISOpticalFlow \"MEDIUM\" Parameters:" << "\n";
    std::cout << "Finest Scale(theta_sf): " << DIS_basic->getFinestScale() << "\n";                                                    // PRESET_MEDIUM:1
    std::cout << "Patch Size(theta_ps): " << DIS_basic->getPatchSize() << "\n";
    std::cout << "Patch Stride(theta_ov): " << DIS_basic->getPatchStride() << "\n";                                                    // PRESET_MEDIUM:3
    std::cout << "Gradient Descent Iterations(theta_it): " << DIS_basic->getGradientDescentIterations() << "\n";                       // PRESET_MEDIUM:25
    std::cout << "Variational Refinement Iterations(theta_vo, theta_vi): " << DIS_basic->getVariationalRefinementIterations() << "\n"; // PRESET_MEDIUM:5
    std::cout << "Variational Refinement Delta(theta_delta): " << DIS_basic->getVariationalRefinementDelta() << "\n";
    std::cout << "Variational Refinement Gamma(theta_gamma): " << DIS_basic->getVariationalRefinementGamma() << "\n";                   // PRESET_MEDIUM:10
    std::cout << "Variational Refinement Alpha(theta_alpha): " << DIS_basic->getVariationalRefinementAlpha() << "\n";                   // PRESET_MEDIUM:10
    DIS_basic->calc(gray0, gray1, flow_basic);
*/

/*
    std::string img0_path("../../experiments/images/frame_0021.png");
    std::string img1_path("../../experiments/images/frame_0022.png");

    cv::Mat img0 = cv::imread(img0_path);
    cv::Mat img1 = cv::imread(img1_path);

    if (img0.empty())
    {
        std::cout << "Can't open image0! Check file path.\n";
        return -1;
    }

    cv::cvtColor(img0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);

    DIS_adjust->calc(gray0, gray1, flow_adjust);

    frame_adjust = flowToColor(flow_adjust);

    cv::imshow("ground truth", frame_gt);
    cv::imshow("adjust", frame_adjust);
    cv::waitKey(0);

    std::cout << "adjust gt mepe: " << meanEndPointError(flow_adjust, flow_gt) << "\n";
*/
