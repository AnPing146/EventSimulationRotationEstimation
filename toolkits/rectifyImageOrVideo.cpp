#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "Player.h"

using namespace std;
// using namespace cv;

int main(int argc, char **argv)
{
    // 同時改正一對相機影像與事件影像
    //  ./toolkits/rectifyImageOrVideo i ./cam_to_cam_thun_00_a.yaml ./000035.png
    //  ./toolkits/rectifyImageOrVideo v ./cam_to_cam_thun_00_a.yaml ./thun_00_a.mp4
    //
    if (argc != 4)
    {
        std::cout << "***\nHints:\n"
                  << std::setw(6) << "" << "argrment[1]: input \"i\" for image, or \"v\" for video\n"
                  << std::setw(6) << "" << "argument[2]: calibration file path\n"
                  << std::setw(6) << "" << "argrment[3]: input file path\n***\n";

        return -1;
    }

    ////
    ////
    //// 1. read camera parameter
    cv::FileStorage fs(argv[2], cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Can't open .yaml file. Check file format and path." << endl;
        return -1;
    }

    // cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, Rotation, Translation
    cv::FileNode n0, n1;

    // intrinsics
    cv::Mat cam0_intrinsic, cam0_distCoeff(1, 4, CV_64FC1);
    cv::Mat cam1_intrinsic, cam1_distCoeff(1, 4, CV_64FC1);
    n0 = fs["intrinsics"]["cam0"];
    n1 = fs["intrinsics"]["camRect1"];
    cam0_intrinsic = cv::Mat::zeros(3, 3, CV_64FC1);
    cam1_intrinsic = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int i = 0; i < 4; i++)
    {
        if (i < 2)
        {
            cam0_intrinsic.at<double>(i, i) = (double)n0["camera_matrix"][i];
            cam1_intrinsic.at<double>(i, i) = (double)n1["camera_matrix"][i];
        }
        else if (i == 2)
        {
            cam0_intrinsic.at<double>(0, i) = (double)n0["camera_matrix"][i];
            cam1_intrinsic.at<double>(0, i) = (double)n1["camera_matrix"][i];
        }
        else
        {
            cam0_intrinsic.at<double>(1, 2) = (double)n0["camera_matrix"][i];
            cam1_intrinsic.at<double>(1, 2) = (double)n1["camera_matrix"][i];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        cam0_distCoeff.at<double>(0, i) = (double)n0["distortion_coeffs"][i];
        cam1_distCoeff.at<double>(0, i) = 0.0;
    }

    // extrinsics
    cv::Mat r1_0(3, 3, CV_64FC1), r0_1rect(3, 3, CV_64FC1), r_rect1(3, 3, CV_64FC1), t1_0(3, 1, CV_64FC1), t0_1rect(3, 1, CV_64FC1);
    n0 = fs["extrinsics"]["T_10"];
    for (int i = 0; i < 3; i++)
    {
        t1_0.at<double>(0, i) = (double)n0[i][3];
        for (int ii = 0; ii < 3; ii++)
        {
            r1_0.at<double>(i, ii) = (double)n0[i][ii];
        }
    }
    n0 = fs["extrinsics"]["R_rect1"];
    for (int i = 0; i < 3; i++)
    {
        for (int ii = 0; ii < 3; ii++)
        {
            r_rect1.at<double>(i, ii) = (double)n0[i][ii];
        }
    }
    r0_1rect = r1_0.t() * r_rect1; // cv::transpose(r_rect1*r1_0, r0_1rect);
    t0_1rect = r_rect1 * (-r1_0.t() * t1_0);
    fs.release();
    cout << "camera0 intrinsics: " << cam0_intrinsic << endl;
    cout << "camera0 disCoeff: " << cam0_distCoeff << endl;
    cout << "camera1Rectify intrinsics: " << cam1_intrinsic << endl;
    cout << "camera1Rectify disCoeff: " << cam1_distCoeff << endl;
    cout << "rotation_1_0: " << r1_0 << endl;
    cout << "translation_1_0: " << t1_0 << endl;
    cout << "\nrotation_0_1rect: " << r0_1rect << endl;
    cout << "translation_0_1rect: " << t0_1rect << endl;
    cout << "R_rect1: " << r_rect1 << endl;
    cout << "---camera parameters loaded---" << endl;

    ////
    ////
    //// 2. read target file, calculate remap, then save rectify result
    cv::Mat r1, r2, p1, p2, q, map[2][2];
    cv::Mat imageFrame, eventFrame, result;
    int frameRows, frameCols;

    if (string(argv[1]) == "i")
    {
        // file read
        imageFrame = cv::imread(string(string(argv[3]).replace(0, 1, "./images/frames")));
        eventFrame = cv::imread(string(string(argv[3]).replace(0, 1, "./images/events")));
        if (imageFrame.empty() || eventFrame.empty())
        {
            cout << "fail to open image, check image file path!\n";
            return -1;
        }
        frameRows = imageFrame.rows;
        frameCols = imageFrame.cols;

        // camera0 intrinsics scalling
        cam0_intrinsic.at<double>(1, 1) *= (double)frameRows / eventFrame.rows; // fy
        cam0_intrinsic.at<double>(0, 0) *= (double)frameCols / eventFrame.cols; // fx
        cam0_intrinsic.at<double>(0, 2) *= (double)frameCols / eventFrame.cols; // cx
        cam0_intrinsic.at<double>(1, 2) *= (double)frameRows / eventFrame.rows; // cy
        cout << "camera0 intrinsics after scale: " << cam0_intrinsic << "\n\n";
        cv::resize(eventFrame, eventFrame, cv::Size(frameCols, frameRows));
        // cout<<imageFrame.size()<<endl;
        // cout<<eventFrame.size()<<endl;
        
        //  rectify with cam0 and cam1Rect
        cv::stereoRectify(cam0_intrinsic, cam0_distCoeff, cam1_intrinsic, cam1_distCoeff, cv::Size(frameCols, frameRows),
                          r0_1rect, t0_1rect, r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0); // , cv::CALIB_ZERO_DISPARITY, 0
        cv::initUndistortRectifyMap(cam0_intrinsic, cam0_distCoeff, r1, p1, cv::Size(frameCols, frameRows), CV_32FC1,
                                    map[0][0], map[0][1]);
        cv::initUndistortRectifyMap(cam1_intrinsic, cam1_distCoeff, r2, p2, cv::Size(frameCols, frameRows), CV_32FC1,
                                    map[1][0], map[1][1]);
        cout << "r1: " << r1 << "\np1: " << p1 << "\n\n";

        cv::hconcat(eventFrame, imageFrame, result);
        for (int i = 0; i < frameRows; i += 40)
        {
            cv::line(result, cv::Point(0, i), cv::Point(frameCols * 2 - 1, i), cv::Scalar(0, 255, 0), 1);
        }
        cv::namedWindow("before rectify");
        cv::imshow("before rectify", result);
        cv::imwrite("rectifyBefore.png", result);
        cv::waitKey(0);
    
        // remap
        cv::remap(eventFrame, eventFrame, map[0][0], map[0][1], cv::INTER_CUBIC);
        cv::remap(imageFrame, imageFrame, map[1][0], map[1][1], cv::INTER_CUBIC);

        cv::hconcat(eventFrame, imageFrame, result);
        for (int i = 0; i < frameRows; i += 40)
        {
            cv::line(result, cv::Point(0, i), cv::Point(frameCols * 2 - 1, i), cv::Scalar(0, 255, 0), 1);
        }
        cv::namedWindow("result");
        cv::imshow("result", result);
        cv::imwrite("./rectifyImage/events/rectifyEvent.png", eventFrame);
        cv::imwrite("./rectifyImage/frames/rectifyFrame.png", imageFrame);
        cv::imwrite("./rectifyImage/rectifyAfter.png", result);
        cv::waitKey(0);

        cout << "images saved!" << endl;
    }
    else if (string(argv[1]) == "v")
    {
        cv::VideoCapture capEvent, capFrame;
        string eventsPath(argv[3]), framesPath(argv[3]);
        size_t strSize = eventsPath.size();

        // file read
        eventsPath.replace(strSize - 4, 1, "_event.");
        capEvent.open(eventsPath);
        cout << eventsPath << endl;
        framesPath.replace(strSize - 4, 1, "_video.");
        capFrame.open(framesPath);
        cout << framesPath << endl;

        if (!capFrame.isOpened() || !capEvent.isOpened())
        {
            cout << "fail to open video, check video file path!\n";
            return -1;
        }
        capFrame >> imageFrame;
        capEvent >> eventFrame;
        frameRows = imageFrame.rows;
        frameCols = imageFrame.cols;

        // camera0 intrinsics scalling
        cam0_intrinsic.at<double>(1, 1) *= (double)frameRows / eventFrame.rows; // fy
        cam0_intrinsic.at<double>(0, 0) *= (double)frameCols / eventFrame.cols; // fx
        cam0_intrinsic.at<double>(0, 2) *= (double)frameCols / eventFrame.cols; // cx
        cam0_intrinsic.at<double>(1, 2) *= (double)frameRows / eventFrame.rows; // cy
        cout << "camera0 intrinsics after scale: " << cam0_intrinsic << "\n\n";
        cv::resize(eventFrame, eventFrame, cv::Size(frameCols, frameRows));

        // cout<<imageFrame.size()<<endl;
        // cout<<eventFrame.size()<<endl;

        // rectify with cam0 and cam1Rect
        cv::stereoRectify(cam0_intrinsic, cam0_distCoeff, cam1_intrinsic, cam1_distCoeff, cv::Size(frameCols, frameRows),
                          r0_1rect, t0_1rect, r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 0); // , cv::CALIB_ZERO_DISPARITY, 0
        cv::initUndistortRectifyMap(cam0_intrinsic, cam0_distCoeff, r1, p1, cv::Size(frameCols, frameRows), CV_32FC1,
                                    map[0][0], map[0][1]);
        cv::initUndistortRectifyMap(cam1_intrinsic, cam1_distCoeff, r2, p2, cv::Size(frameCols, frameRows), CV_32FC1,
                                    map[1][0], map[1][1]);
        cout << "r1: " << r1 << "\np1: " << p1 << "\n\n";

        // remap first image for present
        cv::remap(eventFrame, eventFrame, map[0][0], map[0][1], cv::INTER_CUBIC);
        cv::remap(imageFrame, imageFrame, map[1][0], map[1][1], cv::INTER_CUBIC);

        cv::hconcat(eventFrame, imageFrame, result);
        for (int i = 0; i < frameRows; i += 40)
        {
            cv::line(result, cv::Point(0, i), cv::Point(frameCols * 2 - 1, i), cv::Scalar(0, 255, 0), 1);
        }
        cv::namedWindow("result");
        cv::imshow("result", result);
        cv::waitKey(0);

        // remap all image, and write rectify video
        cv::VideoWriter eventsWriter, framesWriter;
        auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

        strSize = eventsPath.size();
        eventsPath.replace(strSize - 4, 1, "_rectify.");
        framesPath.replace(strSize - 4, 1, "_rectify.");
        // cout << eventsPath << endl;
        // cout << framesPath << endl;

        eventsWriter.open(eventsPath, fourcc, 20, cv::Size(frameCols, frameRows));
        framesWriter.open(framesPath, fourcc, 20, cv::Size(frameCols, frameRows));

        int frameSize = (int)capFrame.get(cv::CAP_PROP_FRAME_COUNT);
        int frameCount = 0;
        if (eventsWriter.isOpened() && framesWriter.isOpened())
        {
            eventsWriter << eventFrame;
            framesWriter << imageFrame;
            frameCount++;
            for (int i = 0; i < frameSize; i++)
            {
                capEvent >> eventFrame;
                if (eventFrame.empty() || imageFrame.empty())
                    break; // 事件莫名會少一張

                cv::resize(eventFrame, eventFrame, cv::Size(frameCols, frameRows));
                cv::remap(eventFrame, eventFrame, map[0][0], map[0][1], cv::INTER_CUBIC);
                eventsWriter << eventFrame;

                capFrame >> imageFrame;
                cv::remap(imageFrame, imageFrame, map[1][0], map[1][1], cv::INTER_CUBIC);
                framesWriter << imageFrame;
                frameCount++;
            }
        }
        else
        {
            cout << "can't open video writer!\n";
        }

        capEvent.release();
        capFrame.release();
        eventsWriter.release();
        framesWriter.release();
        cout <<"(" <<frameSize << ") count frames: "<<frameCount <<", video saved!" << endl;

    }
    else
    {
        cout << "fail input, argrment[1]: input \"i\" for image, or \"v\" for video\n";
        return -1;
    }

    return 0;
}

/*
    // camera0 intrinsics scalling
    cam0_intrinsic.at<double>(0, 0) *= (double)frameSize.width / eventSize.width;   // fx
    cam0_intrinsic.at<double>(1, 1) *= (double)frameSize.height / eventSize.height; // fy
    cam0_intrinsic.at<double>(0, 2) *= (double)frameSize.width / eventSize.width;   // cx
    cam0_intrinsic.at<double>(1, 2) *= (double)frameSize.height / eventSize.height; // cy
    cout << "camera0 intrinsics after scale: " << cam0_intrinsic <<"\n\n";
    // cout<<"x scale: "<<frameSize.width/eventSize.width<<", y scale: "<<frameSize.height/eventSize.height<<endl;
    // cout<<"x scale: "<<(double)frameSize.width/eventSize.width<<", y scale: "<<(double)frameSize.height/eventSize.height<<endl;

    // remap
    cv::Mat imageFrame, eventFrame, result;
    int frameRows, frameCols;
    imageFrame = cv::imread("./images/frames/000035.png");
    eventFrame = cv::imread("./images/events/000035.png");

    frameRows = imageFrame.rows;
    frameCols = imageFrame.cols;
    cv::resize(eventFrame, eventFrame, cv::Size(frameCols, frameRows));
    //cout<<imageFrame.size()<<endl;
    //cout<<eventFrame.size()<<endl;
    cv::stereoRectify(cam0_intrinsic, cam0_distCoeff, cam1_intrinsic, cam1_distCoeff, cv::Size(frameCols, frameRows),\
                      r0_1rect, t0_1rect, r1, r2, p1, p2, q);       // rectify with cam0 and cam1Rect // , cv::CALIB_ZERO_DISPARITY, 0
    cv::initUndistortRectifyMap(cam0_intrinsic, cam0_distCoeff, r1, p1, cv::Size(frameCols, frameRows), CV_32FC1,\
                                map[0][0], map[0][1]);
    cv::initUndistortRectifyMap(cam1_intrinsic, cam1_distCoeff, r2, p2, cv::Size(frameCols, frameRows), CV_32FC1,\
                                map[1][0], map[1][1]);
    cout<<"r1: "<<r1<<"\np1: "<<p1<<"\n";

    cv::hconcat(eventFrame, imageFrame, result);
    for (int i = 0; i < frameRows; i += 40)
    {
        cv::line(result, cv::Point(0, i), cv::Point(frameCols * 2 - 1, i), cv::Scalar(0, 255, 0), 1);
    }
    cv::namedWindow("before rectify");
    cv::imshow("before rectify", result);
    cv::imwrite("rectifyBefore.png", result);
    cv::waitKey(0);

    cv::remap(eventFrame, eventFrame, map[0][0], map[0][1], cv::INTER_CUBIC);
    cv::remap(imageFrame, imageFrame, map[1][0], map[1][1], cv::INTER_CUBIC);

    cv::hconcat(eventFrame, imageFrame, result);
    for (int i = 0; i < frameRows; i += 40)
    {
        cv::line(result, cv::Point(0, i), cv::Point(frameCols * 2 - 1, i), cv::Scalar(0, 255, 0), 1);
    }
    cv::namedWindow("result");
    cv::imshow("result", result);
    cv::imwrite("rectifyAfter.png", result);
    cv::waitKey(0);

*/
