#include "System.h"
#include "numerics.h"

// #include <Eigen/Core>

#include <opencv2/core.hpp>
// #include <opencv2/core/eigen.hpp> //依賴外部配置Eigen! 要先引入eigen再引入opencv/eigen.hpp

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
//#include <deque>

System::System(const std::string &yaml)
{
    std::cout << "\n"
              << "RME-EC: Rotational Motion Estimation with Event Camera" << "\n"
              << "\n"
                 "Copyright (C) 2020 Haram Kim"
              << "\n"
              << "Lab for Autonomous Robotics Research, Seoul Nat'l Univ." << "\n"
              << "\n";

    // Read settings file
    cv::FileStorage fsSettings(yaml, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << yaml << std::endl;
        std::exit(-1);
    }

    // cv::namedWindow("map image", CV_WINDOW_AUTOSIZE);
    // cv::namedWindow("warped event image", CV_WINDOW_AUTOSIZE);
    // cv::namedWindow("undistorted renderer", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("map image view", CV_WINDOW_AUTOSIZE);

    /// Set parameters

    // general params
    plot_denom = fsSettings["visualize.denom"];
    sampling_rate = fsSettings["sampling_rate"];
    map_sampling_rate = fsSettings["map_sampling_rate"];
    sliding_window_size = static_cast<uint32_t>((int)fsSettings["sliding_window_size"]);
    event_image_threshold = fsSettings["event_image_threshold"];
    rotation_estimation = (int)fsSettings["rotation_estimation"];
    run_index = fsSettings["run_index"];
    mapping_interval = fsSettings["mapping_interval"];
    optimization_view_index = fsSettings["optimization_view_index"];
    optimization_view_flag = false;

    // event params
    delta_time = fsSettings["delta_time"];
    max_event_num = fsSettings["max_event_num"];

    // camera params
    width = fsSettings["width"];
    height = fsSettings["height"];

    map_scale = fsSettings["map_scale"];

    camParam.fx = fsSettings["Camera.fx"];
    camParam.fy = fsSettings["Camera.fy"];
    camParam.cx = fsSettings["Camera.cx"];
    camParam.cy = fsSettings["Camera.cy"];
    camParam.rd1 = fsSettings["Camera.rd1"];
    camParam.rd2 = fsSettings["Camera.rd2"];

    if (sampling_rate <= 0)
    {
        std::cerr << "Invalid sampling_rate: " << sampling_rate << std::endl;
        std::exit(-1);
    }

    K = cv::Matx33d(camParam.fx, 0.0, camParam.cx,
                    0.0, camParam.fy, camParam.cy,
                    0.0, 0.0, 1.0);
    distCoeffs = cv::Matx14d(camParam.rd1, camParam.rd2, 0.0, 0.0);

    // grad_x_kernel = (cv::Mat1f(1, 3) << -0.5f, 0.0f, 0.5f);
    // grad_y_kernel = (cv::Mat1f(3, 1) << -0.5f, 0.0f, 0.5f);

    width_map = map_scale * width;
    height_map = map_scale * height;
    K_map = K;
    K_map(0, 2) = width_map / 2;
    K_map(1, 2) = height_map / 2;

    // optimization parameters
    optimizer_max_iter = fsSettings["Optimizer.max_iter"];

    eta_event = fsSettings["Optimizer.eta_angular_velocity"];
    rho_event = fsSettings["Optimizer.rho_angular_velocity"];

    eta_map = fsSettings["Optimizer.eta_angular_position"];
    rho_map = fsSettings["Optimizer.rho_angular_position"];

    /// Initialization
    denominator = round4(1.0f / plot_denom);
    last_event_time = 0.0f;
    estimation_time_interval = -1;
    estimation_time_interval_prev = -1;

    // image
    current_image.image = cv::Mat(height, width, CV_16UC1); // 沒處理實際影像，所以沒用到
    current_image.image = cv::Scalar(0);
    undistorted_image.image = cv::Mat(height, width, CV_16UC1); // 沒處理實際影像，所以沒用到
    undistorted_image.image = cv::Scalar(0);
    warped_event_image = cv::Mat(height, width, CV_32FC1);
    warped_event_image = cv::Scalar(0);
    warped_map_image = cv::Mat(height, width, CV_32FC1);
    warped_map_image = cv::Scalar(0);
    undistorted_render = cv::Mat(height, width, CV_32FC3);
    undistorted_render = cv::Scalar(0, 0, 0);
    map_render = cv::Mat(height_map, width_map, CV_32FC3);
    map_render = cv::Scalar(0, 0, 0);

    Ix_map = cv::Mat(height, width, CV_32FC1);
    Iy_map = cv::Mat(height, width, CV_32FC1);

    GenerateMesh();

    // time
    next_process_time = delta_time;

    // counter
    map_iter_begin = 0;

    // event_data_iter = 0;
    // image_data_iter = 0;
    // imu_data_iter = 0;

    // angular velocity
    angular_velocity = cv::Vec3d::all(0.0);
    angular_position = cv::Vec3d::all(0.0);
    angular_velocity_prev = cv::Vec3d::all(0.0);
    angular_position_prev = cv::Vec3d::all(0.0);
    angular_position_init = cv::Vec3d::all(0.0);

    update_angular_velocity = cv::Vec3d::all(0.0);
    update_angular_position = cv::Vec3d::all(0.0);

    historical_angular_velocity.reserve(1000);
    historical_angular_position.reserve(1000);
    historical_timestamp.reserve(1000);

    std::cout << "rotation_estimation: " << rotation_estimation << "\n";distCoeffs;
    //std::cout << "K: " << K << "\n";distCoeffs;
    //std::cout << "distCoeffs: " << distCoeffs << "\n\n";
    bindCnt = 0; ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cv::waitKey(1000);
}

System::~System()
{
}

void System::PushImageData(ImageData imageData)
{
    vec_image_data.push_back(imageData);
    current_image = imageData;
    UndistortImage();
}

void System::PushImuData(ImuData imuData)
{
    vec_imu_data.push_back(imuData);
}

/*
//因為不需要轉換ROS msg，所以不需要用vec_event_data再多轉一次才變event
void System::PushEventData(EventData eventData){
    vec_event_data.push_back(eventData);
    if(run_index){
        RunUptoIdx();
    }
    else{
        Run();
    }
}
*/
void System::BindEvent(std::vector<Event> &eventData)
{
    std::size_t sampled_count = 0;
    std::size_t total_events = eventData.size();
    bindCnt++;

    // 預先分配空間
    eventBundle.x.reserve(eventBundle.x.size() + total_events / sampling_rate);
    eventBundle.y.reserve(eventBundle.y.size() + total_events / sampling_rate);
    eventBundle.time_stamp.reserve(eventBundle.time_stamp.size() + total_events / sampling_rate);
    eventBundle.polarity.reserve(eventBundle.polarity.size() + total_events / sampling_rate);

    Event *ptr = eventData.data();
    Event *end = ptr + total_events;

    // 寫入資料
    while (ptr < end)
    {
        if (sampled_count % sampling_rate == 0)
        {
            eventBundle.x.push_back(ptr->x);
            eventBundle.y.push_back(ptr->y);
            eventBundle.time_stamp.push_back(ptr->timestamp);
            eventBundle.polarity.push_back(ptr->polarity);
        }
        ++sampled_count;
        ++ptr;
    }

    eventBundle.size = eventBundle.x.size(); // 更新 size

    if(eventBundle.size != 0){
        if (run_index)
        {
            RunUptoIdx();
        }
        else
        {
            Run();
        }
    }
}

void System::BindMap()
{
    eventMap.Clear();
    //std::cout<<"before eventMap.size(): "<<eventMap.time_stamp.size()<<" ***\n";
    //std::cout<<"before map_iter_begin: "<<map_iter_begin<<" ***\n";
    // EventBundle temp;             //在這邊沒用到
    for (uint16_t m_iter = map_iter_begin; m_iter != eventMapPoints.size(); m_iter++)
    {
        EventBundle &curEvent = eventMapPoints.at(m_iter);
        curEvent.size = curEvent.coord_3d.size();
        //排除與目前追蹤角度差距太大的「落後事件」或「非穩定狀態」的事件包
        if (curEvent.size == 0 || cv::norm(SO3add(curEvent.angular_position, -angular_position_prev)) > mapping_interval)
        {
            map_iter_begin = m_iter + 1;    //紀錄已挑選過的全域對齊事件包
            //std::cout<<"map_iter_begin: "<<map_iter_begin<<" ***\n";
            continue;
        }
        eventMap.Append(curEvent);
    }
    //std::cout<<"after map_iter_begin: "<<map_iter_begin<<" ***\n";
    //std::cout<<"eventMapPoints.size(): "<<eventMapPoints.size()<<" ***\n";
    eventWarpedMap.Copy(eventMap); // copy()缺少角度資訊，但後續GetWarpedEventPoint()時會補上角度數據
    std::cout<<"eventWarpedMap.size(): "<<eventWarpedMap.time_stamp.size()<<"\n";
}

void System::ClearEvent()
{
    eventBundle.Clear();
    eventUndistorted.Clear();
    eventWarped.Clear();
}

// run the algorithm
void System::Run()
{
    // double time_begin = ros::Time::now().toSec();   //沒用到
    // BindEvent();
    // 原作者步驟是 System::PushEventData(...) => System::Run() => System::BindEvent(...)
    // 但改成直接 System::BindEvent(event...) => System::Run()

    if (eventBundle.time_stamp.size() == 0) 
    {
        return;
    }
    double event_interval = eventBundle.time_stamp.back() - eventBundle.time_stamp.front();

    if (eventBundle.time_stamp.back() >= next_process_time || eventBundle.x.size() > max_event_num)
    {
        if(eventBundle.time_stamp.back() >= next_process_time){ //檢查以哪種條件進入運算
            std::cout<<"*** eventBundle.timestamp(): "<<eventBundle.time_stamp.back()<<" ***\n";
            std::cout<<"*** eventBundle.size(): "<<eventBundle.x.size()<<" ***\n\n";
        }else{
            std::cout<<"*** eventBundle.size(): "<<eventBundle.x.size()<<" ***\n\n";
        }

        next_process_time = eventBundle.time_stamp.back() + delta_time;
        eventBundle.SetCoord();
        EstimateMotion();
        Renderer();
        Visualize();
        ClearEvent();
    }

    /*
    // for dataset repeat
    //在ROS中訊息重播時重設next_process_time
    if (vec_event_data.back().time_stamp < delta_time)
    {
        next_process_time = delta_time;
    }
     */
}

void System::RunUptoIdx()
{

    // BindEvent();          改成直接 System::BindEvent(event...) => System::Run()
    if (eventBundle.time_stamp.size() == 0)
    {
        return;
    }

    if (eventBundle.time_stamp.back() >= next_process_time || eventBundle.x.size() > max_event_num)
    {
        next_process_time = eventBundle.time_stamp.back() + delta_time;
        vec_event_bundle_data.push_back(eventBundle);
        ClearEvent();
    }

    // main loop
    if (vec_event_bundle_data.size() > run_index)
    {
        for (size_t iter = 0; iter < vec_event_bundle_data.size(); iter++)
        {
            if (iter > optimization_view_index)
            {
                optimization_view_flag = true;
            }
            eventBundle = vec_event_bundle_data[iter];
            // double time_begin = ros::Time::now().toSec();   //沒用到
            double event_interval = eventBundle.time_stamp.back() - eventBundle.time_stamp.front();
            eventBundle.SetCoord();
            EstimateMotion();
            Renderer();
            Visualize();
        }
        next_process_time = 0.0f;
        cv::waitKey(0); // 暫停？？？
    }
}

void System::Visualize()
{
    // imshow("map image", warped_map_image * denominator);
    // imshow("warped event image", warped_event_image * denominator);
    // imshow("undistorted renderer", undistorted_render);
    imshow("map image view", map_render);
    cv::waitKey(1);
}

void System::Renderer()
{
    map_render = GetMapImageNew(SIGNED_EVENT_IMAGE_COLOR).clone();
}

// Get event image from various option
cv::Mat System::GetEventImage(const EventBundle &event, const PlotOption &option, const bool &is_large_size /* = false*/)
{
    int height, width;
    int x, y;
    if (is_large_size)
    {
        height = this->height_map;
        width = this->width_map;
    }
    else
    {
        height = this->height;
        width = this->width;
    }
    cv::Mat event_image;
    switch (option)
    {
    case DVS_RENDERER:
        if (undistorted_image.image.rows != 0 && undistorted_image.image.cols != 0)
        {
            cv::cvtColor(undistorted_image.image, event_image, CV_GRAY2BGR);
        }
        else
        {
            event_image = cv::Mat(height, width, CV_8UC3);
            event_image = cv::Scalar(0, 0, 0);
        }
        for (int i = 0; i < event.size; i++)
        {
            if (!event.isInner[i])
            {
                continue;
            }
            x = event.x[i];
            y = event.y[i];
            event_image.at<cv::Vec3b>(cv::Point(x, y)) = (event.polarity[i] == true ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 0, 255));
        }
        break;
    case SIGNED_EVENT_IMAGE:
        event_image = cv::Mat(height, width, CV_16SC1);
        event_image = cv::Scalar(0);
        for (int i = 0; i < event.size; i++)
        {
            if (!event.isInner[i])
            {
                continue;
            }
            x = event.x[i];
            y = event.y[i];
            event_image.at<short>(cv::Point(x, y)) += (event.polarity[i] == true ? 1 : -1);
        }
        break;
    case SIGNED_EVENT_IMAGE_COLOR:
        event_image = cv::Mat(height, width, CV_16UC3);
        event_image = cv::Scalar(0, 0, 0);
        for (int i = 0; i < event.size; i++)
        {
            if (!event.isInner[i])
            {
                continue;
            }
            x = event.x[i];
            y = event.y[i];
            event_image.at<cv::Vec3w>(cv::Point(x, y)) += (event.polarity[i] == true ? cv::Vec3b(1, 0, 0) : cv::Vec3b(0, 0, 1));
        }
        break;
    case UNSIGNED_EVENT_IMAGE:
        event_image = cv::Mat(height, width, CV_16UC1);
        event_image = cv::Scalar(0);
        for (int i = 0; i < event.size; i++)
        {
            if (!event.isInner[i])
            {
                continue;
            }
            x = event.x[i];
            y = event.y[i];
            event_image.at<unsigned short>(cv::Point(x, y)) += 1;
        }
        break;
    case GRAY_EVENT_IMAGE:
        event_image = cv::Mat(height, width, CV_32FC1);
        event_image = cv::Scalar(0.5);
        for (int i = 0; i < event.size; i++)
        {
            if (!event.isInner[i])
            {
                continue;
            }
            x = event.x[i];
            y = event.y[i];
            event_image.at<float>(cv::Point(x, y)) += (event.polarity[i] == true ? 0.05 : -0.05);
        }
        break;
    }

    return event_image;
}


/*
void System::BindEvent(const std::vector<Event> &eventData)
{
    // 預先分配空間
    std::cout<<"sampling_rate: "<<sampling_rate<<"\n";

    int sampled_count = 0;
    for (const auto &iter : eventData)
    {
        if (sampled_count % sampling_rate == 0)
        {
            ++sampled_count;
        }
    }

    eventBundle.x.resize(eventBundle.x.size() + sampled_count);
    eventBundle.y.resize(eventBundle.y.size() + sampled_count);
    eventBundle.time_stamp.resize(eventBundle.time_stamp.size() + sampled_count);
    eventBundle.polarity.resize(eventBundle.polarity.size() + sampled_count);

    // 寫入資料
    std::size_t offset = eventBundle.x.size() - sampled_count;
    std::cout<<"sampled_count: "<<sampled_count<<"\n";

    std::size_t j = 0;
    for (const auto &iter : eventData)
    {
        if ((j++) % sampling_rate != 0){
            continue;
        }

        std::size_t idx = offset++;
        eventBundle.x[idx] = iter.x;
        eventBundle.y[idx] = iter.y;
        eventBundle.time_stamp[idx] = iter.timestamp;
        eventBundle.polarity[idx] = iter.polarity;
    }

    std::cout<<"eventData.size: "<<eventData.size()<<"\n\n";

    std::cout<<"eventBundle.size: "<<eventBundle.size<<"\n";
    std::cout<<"eventBundle.x.size: "<<eventBundle.x.size()<<"\n";
    std::cout<<"eventBundle.y.size: "<<eventBundle.y.size()<<"\n";
    std::cout<<"eventBundle.time_stamp.size: "<<eventBundle.time_stamp.size()<<"\n";
    std::cout<<"eventBundle.polarity.size: "<<eventBundle.polarity.size()<<"\n\n";

    std::cout<<"eventBundle.coord_2d.size: "<<eventBundle.coord.size()<<"\n";
    std::cout<<"eventBundle.coord_3dsize: "<<eventBundle.coord_3d.size()<<"\n";
    std::cout<<"eventBundle.time_delta.size: "<<eventBundle.time_delta.size()<<"\n";


    if (run_index)
    {
        RunUptoIdx();
    }

    else
    {
        Run();
    }
}
*/

/*
void System::BindEvent(std::vector<Event> &eventData)
{
    // const EventData & eventData = vec_event_data[event_data_iter];
    // std::vector<dvs_msgs::Event> data = eventData.event;
    int sampler = 0; // 取樣基數
    for (std::vector<Event>::iterator it = eventData.begin(); it != eventData.end(); it++)
    {
        if (sampler++ % sampling_rate != 0)
        {
            continue;
        }
        eventBundle.x.push_back(it->x);
        eventBundle.y.push_back(it->y);
        eventBundle.time_stamp.push_back(it->timestamp);
        eventBundle.polarity.push_back(it->polarity);
    }
    // event_data_iter++;

    if (run_index)
    {
        RunUptoIdx();
    }
    else
    {
        Run();
    }
}
*/