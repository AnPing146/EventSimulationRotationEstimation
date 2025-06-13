#include "System.h"
//#include <Eigen/Core>
//#include <opencv2/core/eigen.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/calib3d.hpp>

void System::UndistortEvent()
{
    if (eventBundle.size == 0){
        return;
    }

    eventUndistorted.Copy(eventBundle);

    // 組成原始點vector（用於undistortPoints）
    std::vector<cv::Point2f> raw_points(eventBundle.size), temp_undistorted_points(eventBundle.size);;
    {
        const float* x_ptr = eventBundle.x.data();
        const float* y_ptr = eventBundle.y.data();
        cv::Point2f* raw_ptr = raw_points.data();

        for (std::size_t i = 0; i < eventBundle.size; ++i)
        {
            raw_ptr[i].x = x_ptr[i];
            raw_ptr[i].y = y_ptr[i];
        }
    }

    // 稀疏點校正
    cv::undistortPoints(raw_points, temp_undistorted_points, K, distCoeffs, cv::noArray(), K);

    // 將已校正點複製到eventUndistorted.coord
    eventUndistorted.coord.resize(eventBundle.size);
    {
        const cv::Point2f* src_ptr = temp_undistorted_points.data();
        cv::Point2f* dst_ptr = eventUndistorted.coord.data();
        std::memcpy(dst_ptr, src_ptr, eventBundle.size * sizeof(cv::Point2f));  // 快速複製
    }

    // coord_2d複製到x-y
    eventUndistorted.x.resize(eventUndistorted.coord.size());
    eventUndistorted.y.resize(eventUndistorted.coord.size());
    {
        const cv::Point2f* coord_ptr = eventUndistorted.coord.data();
        float* x_ptr = eventUndistorted.x.data();
        float* y_ptr = eventUndistorted.y.data();

        for (std::size_t i = 0; i < eventUndistorted.coord.size(); ++i)
        {
            x_ptr[i] = coord_ptr[i].x;
            y_ptr[i] = coord_ptr[i].y;
        }
    }

    eventUndistorted.DiscriminateInner(width, height);
    eventUndistorted.SetCoord();  //投影成三維點
}

// Generate mesh for undistortion
void System::GenerateMesh(){
    cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat::eye(3, 3, CV_32FC1), K, cv::Size(width, height), CV_32FC1, 
                                undist_mesh_x, undist_mesh_y); 
}
// Undistort image
void System::UndistortImage(){
    undistorted_image.time_stamp = current_image.time_stamp;
    undistorted_image.seq = current_image.seq;

    cv::remap(current_image.image, undistorted_image.image, undist_mesh_x, undist_mesh_y, CV_INTER_LINEAR);
}

/*
void System::UndistortEvent(){
    if(eventBundle.size == 0){
        return;
    }
    
    eventUndistorted.Copy(eventBundle);                 //事件包轉point2f，以使用校正函式
    std::vector<cv::Point2f> raw_event_point(eventBundle.size), undistorted_event_point(eventBundle.size);    

    for (size_t pts_iter = 0; pts_iter < eventBundle.size; pts_iter++){
        raw_event_point[pts_iter] = cv::Point2f(eventBundle.x[pts_iter], eventBundle.y[pts_iter]);
    }
    cv::undistortPoints(raw_event_point, undistorted_event_point, K, distCoeffs, cv::noArray(), K); 

    cv::Mat temp = cv::Mat(eventBundle.size, 2, CV_32FC1);  //校正後事件點轉n-by-2 mat，再轉eventUndistorted.coord
    temp.data = cv::Mat(undistorted_event_point).data;  
    cv::cv2eigen(temp, eventUndistorted.coord);
    
    //由eventUndistorted.coord手動輸入xy
    //因為eventUndistorted由eventBundle轉換，xy與coord大小不一定一致，無法直接使用setXY()
    eventUndistorted.x.resize(eventUndistorted.coord.col(0).size());
    eventUndistorted.y.resize(eventUndistorted.coord.col(1).size());

    Eigen::VectorXf::Map(&eventUndistorted.x[0], eventUndistorted.coord.col(0).size()) = Eigen::VectorXf(eventUndistorted.coord.col(0));
    Eigen::VectorXf::Map(&eventUndistorted.y[0], eventUndistorted.coord.col(1).size()) = Eigen::VectorXf(eventUndistorted.coord.col(1));  
    
    eventUndistorted.DiscriminateInner(width, height);
    eventUndistorted.SetCoord();                        //投影到三維
}
*/