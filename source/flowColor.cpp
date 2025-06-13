#include "flowColor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

cv::Mat readFlowFile(const std::string &filename)
{
    const float TAG_FLOAT = 202021.25f; // sintel .flo 檔的標記值

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file, check file name.");

    float tag;
    int width, height;
    file.read(reinterpret_cast<char *>(&tag), sizeof(float));
    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));

    if (tag != TAG_FLOAT)
        throw std::runtime_error("Invalid .flo file tag, 非sintel格式.");

    // 讀取 u-v
    int nElements = 2 * width * height;
    std::vector<float> data(nElements);
    file.read(reinterpret_cast<char *>(data.data()), nElements * sizeof(float));

    cv::Mat flow(height, width, CV_32FC2);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            flow.at<cv::Vec2f>(i, j) = cv::Vec2f(data[2 * (i * width + j)], data[2 * (i * width + j) + 1]);
        }
    }
    file.close();

    return flow;
}

void writeFlowFile(const cv::Mat &flow, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file to write: " + filename);

    const float TAG_FLOAT = 202021.25f; // sintel .flo 檔的標記值
    int width = flow.cols;
    int height = flow.rows;

    // write the header
    file.write(reinterpret_cast<const char *>(&TAG_FLOAT), sizeof(float));
    file.write(reinterpret_cast<const char *>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(&height), sizeof(int));

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            cv::Vec2f val = flow.at<cv::Vec2f>(i, j);
            file.write(reinterpret_cast<const char *>(&val[0]), sizeof(float));
            file.write(reinterpret_cast<const char *>(&val[1]), sizeof(float));
        }
    }
    file.close();
}

cv::Mat flowToColor(const cv::Mat &flow)
{
    if (flow.channels() != 2)
    {
        throw std::runtime_error("Envalid flow channels. 稠密光流應為u-v通道.");
    }

    cv::Mat flow_uv[2];
    cv::Mat mag, ang;
    cv::Mat hsv_split[3], hsv, rgb;

    cv::split(flow, flow_uv);
    cv::multiply(flow_uv[1], -1, flow_uv[1]); // 反轉y軸
    cv::cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = cv::Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

    return rgb;
}

cv::Mat colorTest()
{
    int size = 512;
    cv::Mat ang(size, size, CV_32F);
    cv::Mat mag(size, size, CV_32F);

    // 建立座標中心
    float cx = size / 2.0f, cy = size / 2.0f;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            float dx = x - cx;
            float dy = y - cy;
            mag.at<float>(y, x) = sqrt(dx * dx + dy * dy) / (size / 2.0f); // 0~1
            ang.at<float>(y, x) = atan2(dy, dx) * 180.0f / CV_PI;          // -180~180
            if (ang.at<float>(y, x) < 0)
                ang.at<float>(y, x) += 360.0f; // 0~360
        }
    }
    // 限制半徑在圓內
    cv::Mat mask = mag <= 1.0f;

    // 產生 flow 2-channel 影像
    cv::Mat flow(size, size, CV_32FC2);
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            if (mask.at<uchar>(y, x))
                flow.at<cv::Vec2f>(y, x) = cv::Vec2f(mag.at<float>(y, x) * cos(ang.at<float>(y, x) * CV_PI / 180.0f),
                                                     mag.at<float>(y, x) * sin(ang.at<float>(y, x) * CV_PI / 180.0f));
            else
                flow.at<cv::Vec2f>(y, x) = cv::Vec2f(0, 0);
        }
    }

    // 使用 flowToColor 產生色彩圖
    cv::Mat colorImg = flowToColor(flow);
    return colorImg;
}

double meanEndPointError(const cv::Mat &flow_src, const cv::Mat &flow_gt)
{
    double mepe = 0.;
    float u_src, u_gt, v_src, v_gt;

    if (flow_src.size() != flow_gt.size())
    {
        throw std::runtime_error("flow_src and flow_gt size not match!\n");
    }

    for (int i = 0; i < flow_src.rows; i++)
    {
        const cv::Vec2f *ptr_src = flow_src.ptr<cv::Vec2f>(i);
        const cv::Vec2f *ptr_gt = flow_gt.ptr<cv::Vec2f>(i);

        for (int j = 0; j < flow_src.cols; j++)
        {
            u_src = ptr_src[j][0];
            v_src = ptr_src[j][1];
            u_gt = ptr_gt[j][0];
            v_gt = ptr_gt[j][1];

            mepe += std::sqrt(std::pow(u_src - u_gt, 2) + std::pow(v_src - v_gt, 2));
        }
    }
    mepe /= (flow_src.rows * flow_src.cols);

    return mepe;
}

std::vector<cv::Mat> readFrame(const std::string &path, const std::string &filenameExtension)
{
    std::vector<cv::String> filenames;
    std::vector<cv::Mat> frames;

    // 搜尋所有 frame_????.png 檔案
    cv::glob(path + "/frame_*" + filenameExtension, filenames, false);

    // 可選：排序檔名，確保順序正確
    std::sort(filenames.begin(), filenames.end());

    if (filenameExtension == ".png")
    {
        for (const auto &file : filenames)
        {
            cv::Mat img = cv::imread(file, cv::IMREAD_UNCHANGED);
            if (img.empty())
            {
                std::cerr << "Warning: Cannot read. " << file << std::endl;
                continue;
            }
            frames.push_back(img);
        }
    }
    else if (filenameExtension == ".flo")
    {
        for (const auto &file : filenames)
        {
            cv::Mat img = readFlowFile(file);
            if (img.empty())
            {
                std::cerr << "Warning: Cannot read. " << file << std::endl;
                continue;
            }
            frames.push_back(img);
        }
    }else{
        throw std::runtime_error("Envaild filename extension!!\n");
    }

    return frames;
}