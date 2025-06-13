#include "System.h"
#include "numerics.h"
// #include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

void System::EstimateMotion()
{
    UndistortEvent();
    eventUndistorted.InverseProjection(K);
    eventWarped.Copy(eventUndistorted);

    float event_num_weight = static_cast<float>(std::min(eventUndistorted.size * 1e-3 + 1e-4, 1.0));
    if (event_num_weight > 1.0)
        std::cout << "\n    event_num_weight: " << event_num_weight << "!!!!!!!!!!!!!!!!!!!\n\n";
    // std::cout<<"event_num_weight: "<<event_num_weight<<"\n";
    //  std::cout<<"eventUndistorted.time_stamp.front(): "<<eventUndistorted.time_stamp.front()<<"\n";
    //  std::cout<<"last_event_time: "<<last_event_time<<"\n";

    estimation_time_interval = eventUndistorted.time_stamp.front() - last_event_time; // 目前的第一刻時間減掉上一次的第一刻時間
    last_event_time = eventUndistorted.time_stamp.front();
    event_delta_time = eventUndistorted.time_stamp.back() - eventUndistorted.time_stamp.front();

    ComputeAngularPosition();
    BindMap();

    angular_velocity = cv::Vec3d::all(0.0);
    angular_position_compensator = cv::Vec3d::all(0.0);
    nu_event = 1.0; //
    nu_map = 1.0;   //

    double map_norm = 0.0, map_norm_reciprocal = 0.0;
    if (rotation_estimation)
    {
        GetWarpedEventMap(angular_position_init);                                                     // 全域對齊扭轉回當前位置
        cv::threshold(warped_map_image, truncated_map, event_image_threshold, 255, cv::THRESH_TRUNC); // image clamped
        map_norm = cv::norm(truncated_map);                                                           // frobenius norm
        map_norm_reciprocal = 1.0 / std::sqrt(map_norm);                                              // the denominator term of normalization

        // std::cout << "map_norm: " << map_norm << "\n";
        // std::cout << "map_norm_reciprocal: " << map_norm_reciprocal << "\n\n";
        //  分母項多做一次開根號，使分母變小而正規化數值稍微變大，旨在提升後面的梯度值
        cv::GaussianBlur(truncated_map * map_norm_reciprocal, truncated_map, cv::Size(21, 21), 3);
        cv::Sobel(truncated_map, Ix_map, CV_32FC1, 1, 0);
        cv::Sobel(truncated_map, Iy_map, CV_32FC1, 0, 1);
    }

    for (int i = 0; i < optimizer_max_iter; i++)
    {
        // Derive Error Analytic
        is_polarity_on = i < optimizer_max_iter / 2; // 只在前半部分，還沒收斂時計算極性
        DeriveErrAnalytic(angular_velocity, angular_position_compensator);

        // 轉換 warped_image_delta_t 為 cv::Mat 以計算 Gradient = J^T * delta_t
        cv::Mat delta_mat((size_t)warped_image_delta_t.size(), 1, CV_64F, warped_image_delta_t.data());
        // std::cout << "Jacobian: " << Jacobian << "\n\n";
        cv::Mat grad_mat;
        cv::gemm(Jacobian, delta_mat, 1.0, cv::noArray(), 0.0, grad_mat, cv::GEMM_1_T); ////////////////////////除錯////////////////////

        // if (grad_mat.at<double>(0) <1e2)
        //{
        //     std::cout << "Jacobian: " << Jacobian.row(0) << "\n";
        //     std::cin.get();
        // }

        Gradient = cv::Matx31d(grad_mat.at<double>(0), grad_mat.at<double>(1), grad_mat.at<double>(2));
        // std::cout << "bindCnt: " << bindCnt << "\n";
        // std::cout << "Gradient(i): " << "(" << i << "):" << Gradient << "\n";
        //  std::cout<<"grad_mat.at<double>(0): "<<grad_mat.at<double>(0)<<"\n\n";
        //   RMS-prop optimizer
        double grad_norm_sq = Gradient.dot(Gradient); // 梯度大的時候很大！！
        nu_event = rho_event * nu_event + (1.0 - rho_event) * grad_norm_sq;

        double scale = -event_num_weight * eta_event / std::sqrt(nu_event + 1e-8);
        cv::Matx31d grad_scaled = Gradient * scale; // 注意精度轉換
        update_angular_velocity = cv::Vec3d(grad_scaled(0), grad_scaled(1), grad_scaled(2));
        angular_velocity = SO3add(update_angular_velocity, angular_velocity, true);

        // R_upd
        if (map_norm != 0 && rotation_estimation)
        {
            cv::Mat delta_map((size_t)warped_map_delta_t.size(), 1, CV_64F, warped_map_delta_t.data());
            cv::Mat grad_map;
            cv::gemm(Jacobian_map, delta_map, 1.0, cv::noArray(), 0.0, grad_map, cv::GEMM_1_T);
            Gradient_map = cv::Matx31d(grad_map.at<double>(0), grad_map.at<double>(1), grad_map.at<double>(2));
            // std::cout << "Gradient_map(i): " << "(" << i << "):" << Gradient_map << "\n";

            // RMS-prop optimizer
            double grad_map_norm_sq = Gradient_map.dot(Gradient_map); // 梯度大的時候很大！！
            nu_map = rho_map * nu_map + (1.0 - rho_map) * grad_map_norm_sq;

            double scale_map = -event_num_weight * eta_map / std::sqrt(nu_map + 1e-8);
            cv::Matx31d grad_map_scaled = Gradient_map * scale_map; // 注意精度轉換
            update_angular_position = cv::Vec3d(grad_map_scaled(0), grad_map_scaled(1), grad_map_scaled(2));

            angular_position_compensator = SO3add(update_angular_position, angular_position_compensator);
        }
    }
    angular_position = SO3add(angular_position_init, angular_position_compensator);
    GetWarpedEventImage(angular_velocity, SIGNED_EVENT_IMAGE_COLOR);
    
    // Record historical data
    historical_angular_velocity.push_back(angular_velocity);
    historical_angular_position.push_back(angular_position);
    historical_timestamp.push_back(eventUndistorted.time_stamp.front() + 
                                     (eventUndistorted.time_stamp.front() - eventUndistorted.time_stamp.back()) / 2.f);

    imshow("warped_event_image", warped_event_image);
    std::cout << "timestamp: " << eventUndistorted.time_stamp.front() << "\n";
    std::cout << "Gradient(cnt): " << "(" << bindCnt << "):" << Gradient << "\n";        
    std::cout << "Gradient_map(cnt): " << "(" << bindCnt << "):" << Gradient_map << "\n"; 
    std::cout << "z-y-x degree: " << rotationVectorToYPR(angular_position) << "\n\n";
}

void System::ComputeAngularPosition()
{
    // Global events alignment
    eventAlignedPoints.Copy(eventUndistortedPrev); // 前一刻的事件包對齊到目前位置
    GetWarpedEventPoint(eventUndistortedPrev, eventAlignedPoints, angular_velocity, false, angular_position);
    eventAlignedPoints.angular_velocity = angular_velocity_prev; // 儲存前一刻的速度
    eventAlignedPoints.angular_position = angular_position;      // 儲存對齊到的位置
    eventMapPoints.push_back(eventAlignedPoints);
    std::cout << "eventMapPoints.size(): " << eventMapPoints.size() << "\n";

    // initial angular position with angular velocity
    if (!std::isfinite(estimation_time_interval_prev) || estimation_time_interval_prev <= 0)
    {
        // std::cout << "estimation_time_interval_prev: " << estimation_time_interval_prev<<"\n\n";
        std::cout << "estimation_time_interval is nan" << std::endl;
    }
    else
    {
        // 回推初始角位置
        angular_position_init = SO3add(angular_position, -angular_velocity * estimation_time_interval);
    }

    eventUndistortedPrev = eventUndistorted;
    estimation_time_interval_prev = estimation_time_interval;
    angular_velocity_prev = angular_velocity;
    angular_position_pprev = angular_position_prev;
    angular_position_prev = angular_position;
}

// Compute Jacobian matrix
// update Ix, Iy, Jacobian, warped_image_valid
// Jacobian = partial_I(x_k) / partial_w = del_I(x_k) * partial_x_k / partial_w
void System::DeriveErrAnalytic(const cv::Vec3d &temp_ang_vel, const cv::Vec3d &temp_ang_pos)
{
    GetWarpedEventImage(temp_ang_vel, temp_ang_pos); // 計算區域扭轉並對齊影像
    // std::cout << "temp_ang_vel: " << temp_ang_vel << "\n\n";
    // std::cout << "temp_ang_pos: " << temp_ang_pos << "\n\n";
    cv::threshold(warped_event_image, truncated_event, event_image_threshold, 255, 2);

    cv::GaussianBlur(truncated_event, blur_image, cv::Size(5, 5), 1);

    cv::Sobel(blur_image, Ix, CV_32FC1, 1, 0);
    cv::Sobel(blur_image, Iy, CV_32FC1, 0, 1);
    // imshow("cv blur_image", blur_image);cv::waitKey(0);
    // imshow("cv Ix", Ix);cv::waitKey(0);
    // imshow("cv Iy", Iy);cv::waitKey(0);

    if (rotation_estimation)
    {
        float event_norm = 2.0f / static_cast<float>(cv::norm(truncated_event)); // 區域扭轉影像的正規化係數
        truncated_sum = truncated_map + truncated_event * event_norm;            // global-local alignment sum
        Ix_sum = Ix_map + Ix * event_norm;
        Iy_sum = Iy_map + Iy * event_norm;

        // std::cout << "event_norm: " << event_norm << "\n\n";
        //  std::cout << "Ix_sum: " << Ix_sum << "\n\n";
        //  std::cout << "Iy_sum: " << Iy_sum << "\n\n";
        if (optimization_view_flag)
        {
            imshow("truncated_evt", truncated_event * 20);
            imshow("truncated_sum", truncated_sum * 5);
            cv::waitKey(10);
        }
    }

    std::vector<uint16_t> e_valid = GetValidIndexFromEvent(eventWarped);
    // std::cout << "eventWarped.size = " << eventWarped.size << std::endl;
    uint16_t valid_size = e_valid.size();
    // std::cout << "valid_size = " << valid_size << std::endl;

    xp.resize(valid_size);
    yp.resize(valid_size);
    zp.resize(valid_size);
    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    warped_image_delta_t.resize(valid_size);
    Jacobian.create(valid_size, 3, CV_64F);

    Ix_interp_map.resize(valid_size);
    Iy_interp_map.resize(valid_size);
    warped_map_delta_t.resize(valid_size);
    Jacobian_map.create(valid_size, 3, CV_64F);

    // 指標初始化
    float *xp_ptr = xp.data();
    float *yp_ptr = yp.data();
    float *zp_ptr = zp.data();
    float *Ix_ptr = Ix_interp.data();
    float *Iy_ptr = Iy_interp.data();
    float *delta_ptr = warped_image_delta_t.data();
    double *J_ptr = Jacobian.ptr<double>();
    double *Jm_ptr = Jacobian_map.ptr<double>();

    float *Ix_map_ptr = Ix_interp_map.data();
    float *Iy_map_ptr = Iy_interp_map.data();
    float *map_delta_ptr = warped_map_delta_t.data();

    int coord_x;
    int coord_y;
    int idx;
    for (uint16_t i = 0; i < valid_size; i++)
    {
        // 取出視窗內(valid)的事件資訊
        idx = e_valid[i];

        coord_x = std::round(eventWarped.coord[idx].x); // nearest-neighbor interpolation取位置
        coord_y = std::round(eventWarped.coord[idx].y);

        xp_ptr[i] = eventWarped.coord_3d[idx].x;
        yp_ptr[i] = eventWarped.coord_3d[idx].y;
        zp_ptr[i] = eventWarped.coord_3d[idx].z;

        Ix_ptr[i] = Ix.at<float>(coord_y, coord_x); // 插值取出有效值位置的梯度
        Iy_ptr[i] = Iy.at<float>(coord_y, coord_x);
        // std::cout << "coord_x: " << coord_x << "\n";
        // std::cout << "coord_y: " << coord_y << "\n";
        // std::cout << "Ix_ptr[i]: " << Ix_ptr[i] << "\n";
        // std::cout << "Iy_ptr[i]: " << Iy_ptr[i] << "\n\n";

        // 區域jacobian公式中的"Im(x_k, w_m) x delta_t"
        delta_ptr[i] = -eventWarped.time_delta[idx] * warped_event_image.at<float>(coord_y, coord_x);
        if (rotation_estimation)
        {
            Ix_map_ptr[i] = Ix_sum.at<float>(coord_y, coord_x);
            Iy_map_ptr[i] = Iy_sum.at<float>(coord_y, coord_x);

            // 全域jacobian公式中的"(I_L + I_G)"
            map_delta_ptr[i] = -truncated_sum.at<float>(coord_y, coord_x);
        }
    }

    double fx = camParam.fx, fy = camParam.fy;
    for (uint16_t i = 0; i < valid_size; ++i)
    {
        Ix_ptr[i] *= fx;
        Iy_ptr[i] *= fy;

        float xz = xp_ptr[i] / zp_ptr[i];
        float yz = yp_ptr[i] / zp_ptr[i];
        float xz2 = xz * xz;
        float yz2 = yz * yz;
        float xyz = xz * yz;

        // 看起來有點亂，為jacobian公式中的中間兩項相乘
        J_ptr[i * 3 + 0] = -(Ix_ptr[i] * xyz + Iy_ptr[i] * (1.0f + yz2));
        J_ptr[i * 3 + 1] = Ix_ptr[i] * (1.0f + xz2) + Iy_ptr[i] * xyz;
        J_ptr[i * 3 + 2] = -Ix_ptr[i] * yz + Iy_ptr[i] * xz;

        if (rotation_estimation)
        {
            Ix_map_ptr[i] *= fx;
            Iy_map_ptr[i] *= fy;
            // std::cout << "fx: " << fx << "\n";
            // std::cout << "fy: " << fy << "\n";
            // std::cout << "Ix_ptr[0]: " << Ix_ptr[0] << "\n";
            // std::cout << "Iy_ptr[0]: " << Iy_ptr[0] << "\n";
            // std::cout << "Ix_map_ptr[0]: " << Iy_map_ptr[0] << "\n";
            // std::cout << "Iy_map_ptr[0]: " << Iy_map_ptr[0] << "\n\n";

            Jm_ptr[i * 3 + 0] = -(Ix_map_ptr[i] * xyz + Iy_map_ptr[i] * (1.0f + yz2));
            Jm_ptr[i * 3 + 1] = Ix_map_ptr[i] * (1.0f + xz2) + Iy_map_ptr[i] * xyz;
            Jm_ptr[i * 3 + 2] = -Ix_map_ptr[i] * yz + Iy_map_ptr[i] * xz;
        }
    }
}

// warping function
void System::GetWarpedEventPoint(const EventBundle &eventIn,
                                 EventBundle &eventOut,
                                 const cv::Vec3d &temp_ang_vel,
                                 const bool &is_map_warping /*= false*/,
                                 const cv::Vec3d &temp_ang_pos /*= cv::Vec3d::all(0.0)*/)
{
    if (eventIn.size == 0)
    {
        std::cout << "eventIn == 0 , bindCnt: " << bindCnt << "\n";
        // std::cout<<"EventIn.size: "<<eventIn.size<<"\n";
        // std::cout<<"EventIn.x.size: "<<eventIn.x.size()<<"\n";
        // std::cout<<"EventIn.coord_2d.size: "<<eventIn.coord.size()<<"\n";
        // std::cout<<"EventIn.coord_3dsize: "<<eventIn.coord_3d.size()<<"\n";
        // std::cout<<"EventIn.time_delta.size: "<<eventIn.time_delta.size()<<"\n";
        std::cout << "EventIn size is zero" << std::endl;
        return;
    }

    double ang_vel_norm = cv::norm(temp_ang_vel);
    double ang_pos_norm = cv::norm(temp_ang_pos);

    eventOut.coord_3d.resize(eventIn.size);

    // rodrigues formula first-order term
    std::vector<cv::Point3f> x_ang_vel_hat(eventIn.size);
    std::vector<cv::Point3f> x_ang_vel_hat_square(eventIn.size);

    const cv::Point3f *in_ptr = eventIn.coord_3d.data();
    cv::Point3f *hat_ptr = x_ang_vel_hat.data();
    cv::Point3f *hat2_ptr = x_ang_vel_hat_square.data();

    for (int i = 0; i < eventIn.size; ++i)
    {
        // first order
        hat_ptr[i].x = -temp_ang_vel[2] * in_ptr[i].y + temp_ang_vel[1] * in_ptr[i].z;
        hat_ptr[i].y = +temp_ang_vel[2] * in_ptr[i].x - temp_ang_vel[0] * in_ptr[i].z;
        hat_ptr[i].z = -temp_ang_vel[1] * in_ptr[i].x + temp_ang_vel[0] * in_ptr[i].y;

        // second order
        hat2_ptr[i].x = -temp_ang_vel[2] * hat_ptr[i].y + temp_ang_vel[1] * hat_ptr[i].z;
        hat2_ptr[i].y = +temp_ang_vel[2] * hat_ptr[i].x - temp_ang_vel[0] * hat_ptr[i].z;
        hat2_ptr[i].z = -temp_ang_vel[1] * hat_ptr[i].x + temp_ang_vel[0] * hat_ptr[i].y;
    }

    // points warping via second-order approximation with Rodrigues' formula
    // 先做區域對齊，再轉到全域對齊
    const std::vector<float> &dt = is_map_warping ? eventIn.time_delta_reverse : eventIn.time_delta;
    cv::Point3f *out_ptr = eventOut.coord_3d.data();

    if (ang_vel_norm < 1e-8)
    {
        for (int i = 0; i < eventIn.size; ++i)
            out_ptr[i] = in_ptr[i];
    }
    else
    {
        for (int i = 0; i < eventIn.size; ++i)
        {
            float t = dt[i];
            float t2 = 0.5f * t * t;
            out_ptr[i].x = in_ptr[i].x + hat_ptr[i].x * t + hat2_ptr[i].x * t2;
            out_ptr[i].y = in_ptr[i].y + hat_ptr[i].y * t + hat2_ptr[i].y * t2;
            out_ptr[i].z = in_ptr[i].z + hat_ptr[i].z * t + hat2_ptr[i].z * t2;
        }
    }

    // 轉回參考位置，local alignment to global alignment
    if (ang_pos_norm > 1e-8)
    {
        cv::Matx33d R = SO3(temp_ang_pos).t();
        for (int i = 0; i < eventIn.size; ++i)
        {
            const cv::Point3f &p = out_ptr[i];
            out_ptr[i].x = static_cast<float>(R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z);
            out_ptr[i].y = static_cast<float>(R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z);
            out_ptr[i].z = static_cast<float>(R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z);
        }
    }
}

std::vector<uint16_t> System::GetValidIndexFromEvent(const EventBundle &event)
{
    std::vector<uint16_t> valid;
    uint16_t valid_counter = 0;
    for (auto e_iter = event.isInner.begin(); e_iter != event.isInner.end(); e_iter++)
    {
        if (*e_iter)
        {
            valid.push_back(valid_counter);
        }
        valid_counter++;
    }
    return valid;
}

void System::GetWarpedEventImage(const cv::Vec3d &temp_ang_vel, const PlotOption &option) // option = UNSIGNED_EVENT_IMAGE
{
    GetWarpedEventPoint(eventUndistorted, eventWarped, temp_ang_vel);
    eventWarped.Projection(K);
    eventWarped.SetXY();
    eventWarped.DiscriminateInner(width - 1, height - 1);
    GetEventImage(eventWarped, option).convertTo(warped_event_image, CV_32FC1);
}

void System::GetWarpedEventImage(const cv::Vec3d &temp_ang_vel, const cv::Vec3d &temp_ang_pos)
{
    GetWarpedEventPoint(eventUndistorted, eventWarped, temp_ang_vel, false, temp_ang_pos);
    eventWarped.Projection(K);
    eventWarped.SetXY();
    eventWarped.DiscriminateInner(width - 1, height - 1);
    if (is_polarity_on)
    {
        GetEventImage(eventWarped, SIGNED_EVENT_IMAGE).convertTo(warped_event_image, CV_32FC1);
    }
    else
    {
        GetEventImage(eventWarped, UNSIGNED_EVENT_IMAGE).convertTo(warped_event_image, CV_32FC1);
    }
}

void System::GetWarpedEventMap(const cv::Vec3d &temp_ang_pos)
{
    if (eventMap.size == 0)
    {
        return;
    }
    GetWarpedEventPoint(eventMap, eventWarpedMap, cv::Vec3d::all(0.0), true, -temp_ang_pos);
    eventWarpedMap.Projection(K);
    eventWarpedMap.SetXY();
    eventWarpedMap.DiscriminateInner(width - 1, height - 1, map_sampling_rate);
    GetEventImage(eventWarpedMap, UNSIGNED_EVENT_IMAGE, false).convertTo(warped_map_image, CV_32FC1);
}

cv::Mat System::GetMapImage(const PlotOption &option /*= UNSIGNED_EVENT_IMAGE*/)
{
    cv::Mat map_image = cv::Mat(height_map, width_map, CV_16UC3); // 累積的映射結果圖
    map_image = cv::Scalar(0, 0, 0);
    cv::Mat map_image_temp = cv::Mat(height_map, width_map, CV_16UC3);
    EventBundle temp;
    for (uint16_t m_iter = map_iter_begin; m_iter != eventMapPoints.size(); m_iter++)
    {
        EventBundle &curEvent = eventMapPoints.at(m_iter);
        curEvent.size = curEvent.coord_3d.size();
        //相鄰兩次角度差距過大，視為雜訊不使用
        if (curEvent.size == 0 || cv::norm(SO3add(curEvent.angular_position, -angular_position)) > mapping_interval)
        {
            // map_iter_begin = m_iter + 1;
            continue;
        }
        temp.Copy(curEvent);
        GetWarpedEventPoint(curEvent, temp, cv::Vec3d::all(0.0), true, -angular_position);
        temp.Projection(K_map);
        temp.SetXY();
        temp.DiscriminateInner(width_map - 1, height_map - 1, map_sampling_rate);
        GetEventImage(temp, option, true).convertTo(map_image_temp, CV_16UC3); // 其實SIGNED_EVENT_IMAGE_COLOR已經是CV_16UC3
        map_image += map_image_temp;
    }
    cv::Mat result;
    map_image.convertTo(result, CV_32FC3);
    return result;
}

cv::Mat System::GetMapImageNew(const PlotOption &option /*= UNSIGNED_EVENT_IMAGE*/)
{
    cv::Mat result = cv::Mat(height_map, width_map, CV_32FC3); // 映射結果圖
    result = cv::Scalar(0, 0, 0);
    EventBundle eventMapTemp;
    eventMapTemp.Copy(eventMap);
    if (eventMap.size == 0)
    {
        return result;
    }
    GetWarpedEventPoint(eventMap, eventMapTemp, cv::Vec3d::all(0.0), true, -angular_position);
    eventMapTemp.Projection(K_map);
    eventMapTemp.SetXY();
    eventMapTemp.DiscriminateInner(width_map - 1, height_map - 1, map_sampling_rate);
    GetEventImage(eventMapTemp, option, true).convertTo(result, CV_32FC3);
    return result;
}

/*

void System::EstimateMotion(){
    UndistortEvent();
    eventUndistorted.InverseProjection(K);
    eventWarped.Copy(eventUndistorted);

    float event_num_weight = static_cast<float>(std::min(eventUndistorted.size * 1e-3 + 1e-4, 1.0));

    estimation_time_interval = eventUndistorted.time_stamp.front() - last_event_time;
    last_event_time = eventUndistorted.time_stamp.front();  //一直都是0?
    event_delta_time = eventUndistorted.time_stamp.back() - eventUndistorted.time_stamp.front();

    ComputeAngularPosition();
    BindMap();

    angular_velocity = cv::Vec3f::all(0.0);
    angular_position_compensator = cv::Vec3f::all(0.0);
    nu_event = 1.0f;                            //
    nu_map = 1.0f;                              //

    float map_norm = 0.0, map_norm_reciprocal = 0.0;
    if(rotation_estimation){
        GetWarpedEventMap(angular_position_init);                                       //全域對齊扭轉回當前位置
        cv::threshold(warped_map_image, truncated_map, event_image_threshold, 255, 2);  //image clamped
        map_norm = cv::norm(truncated_map);                                             //frobenius norm
        map_norm_reciprocal = 1 / sqrt(map_norm);                                       //the denominator term of normalization
        //分母項多做一次開根號，使分母變小而正規化數值稍微變大，旨在提升後面的梯度值
        cv::GaussianBlur(truncated_map * map_norm_reciprocal, truncated_map, cv::Size(21, 21), 3);
        cv::Sobel(truncated_map, Ix_map, CV_32FC1, 1, 0);
        cv::Sobel(truncated_map, Iy_map, CV_32FC1, 0, 1);
    }

    for (int i = 0; i < optimizer_max_iter; i++){
        // Derive Error Analytic
        is_polarity_on = i < optimizer_max_iter / 2;                        //只在前半部分，還沒收斂時計算極性
        DeriveErrAnalytic(angular_velocity, angular_position_compensator);
        Gradient = Jacobian.transpose() * warped_image_delta_t;
        // RMS-prop optimizer
        nu_event = rho_event * nu_event
                + (1.0f - rho_event) * (float)(Gradient.transpose() * Gradient);
        update_angular_velocity = - event_num_weight * eta_event / std::sqrt(nu_event + 1e-8) * Gradient;
        angular_velocity = SO3add(update_angular_velocity, angular_velocity, true);
        // R_upd
        if(map_norm != 0 && rotation_estimation){
            Gradient_map = Jacobian_map.transpose() * warped_map_delta_t;
            //RMS-prop optimizer
            nu_map = rho_map * nu_map
                    + (1.0f - rho_map) * (float)(Gradient_map.transpose() * Gradient_map);
            update_angular_position = - event_num_weight * eta_map / std::sqrt(nu_map + 1e-8) * Gradient_map;
            angular_position_compensator = SO3add(update_angular_position, angular_position_compensator);
        }
    }
    angular_position = SO3add(angular_position_init, angular_position_compensator);
    GetWarpedEventImage(angular_velocity, SIGNED_EVENT_IMAGE_COLOR);
}

void System::ComputeAngularPosition(){
    // Global events alignment
    eventAlignedPoints.Copy(eventUndistortedPrev);                  //前一刻的事件包對齊到目前位置
    GetWarpedEventPoint(eventUndistortedPrev, eventAlignedPoints, angular_velocity, false, angular_position);
    eventAlignedPoints.angular_velocity = angular_velocity_prev;    //儲存前一刻的速度
    eventAlignedPoints.angular_position = angular_position;         //儲存對齊到的位置
    eventMapPoints.push_back(eventAlignedPoints);

    // initial angular position with angular velocity
    if(!std::isfinite(estimation_time_interval_prev) || estimation_time_interval_prev <= 0){
        std::cout << "estimation_time_interval is nan" << std::endl;
    }
    else{
        //回推初始角位置
        angular_position_init = SO3add(angular_position, -angular_velocity * estimation_time_interval);
    }

    eventUndistortedPrev = eventUndistorted;
    estimation_time_interval_prev = estimation_time_interval;
    angular_velocity_prev = angular_velocity;
    angular_position_pprev = angular_position_prev;
    angular_position_prev = angular_position;
}

// Compute Jacobian matrix
// update Ix, Iy, Jacobian, warped_image_valid
//Jacobian = partial_I(x_k) / partial_w = del_I(x_k) * partial_x_k / partial_w
void System::DeriveErrAnalytic(const Eigen::Vector3f &temp_ang_vel, const Eigen::Vector3f &temp_ang_pos)
{
    GetWarpedEventImage(temp_ang_vel, temp_ang_pos);                            //計算區域扭轉並對齊影像
    cv::threshold(warped_event_image, truncated_event, event_image_threshold, 255, 2);

    cv::GaussianBlur(truncated_event, blur_image, cv::Size(5, 5), 1);

    cv::Sobel(blur_image, Ix, CV_32FC1, 1, 0);
    cv::Sobel(blur_image, Iy, CV_32FC1, 0, 1);

    if(rotation_estimation){
        float event_norm = 2.0f / cv::norm(truncated_event);                    //區域扭轉影像的正規化係數
        truncated_sum = truncated_map + truncated_event * event_norm;           //global-local alignment sum
        Ix_sum = Ix_map + Ix * event_norm;
        Iy_sum = Iy_map + Iy * event_norm;
        if(optimization_view_flag){
            imshow("truncated_evt", truncated_event * 20);
            imshow("truncated_sum", truncated_sum * 5);
            cv::waitKey(10);
        }
    }

    std::vector<uint16_t> e_valid = GetValidIndexFromEvent(eventWarped);

    uint16_t valid_size = e_valid.size();
    xp.resize(valid_size);
    yp.resize(valid_size);
    zp.resize(valid_size);
    Ix_interp.resize(valid_size);
    Iy_interp.resize(valid_size);
    warped_image_delta_t.resize(valid_size);
    Jacobian.resize(valid_size, 3);

    Ix_interp_map.resize(valid_size);
    Iy_interp_map.resize(valid_size);
    warped_map_delta_t.resize(valid_size);
    Jacobian_map.resize(valid_size, 3);

    int16_t coord_x;
    int16_t coord_y;
    uint16_t e_valid_v_iter;
    for (uint16_t v_iter = 0; v_iter < valid_size; v_iter++)
    {
        //取出視窗內(valid)的事件資訊
        e_valid_v_iter = e_valid[v_iter];

        coord_x = std::round(eventWarped.coord(e_valid_v_iter, 0));     //nearest-neighbor interpolation取位置
        coord_y = std::round(eventWarped.coord(e_valid_v_iter, 1));

        xp(v_iter) = eventWarped.coord_3d(e_valid_v_iter, 0);
        yp(v_iter) = eventWarped.coord_3d(e_valid_v_iter, 1);
        zp(v_iter) = eventWarped.coord_3d(e_valid_v_iter, 2);

        Ix_interp(v_iter) = Ix.at<float>(coord_y, coord_x);             //插值取出有效值位置的梯度
        Iy_interp(v_iter) = Iy.at<float>(coord_y, coord_x);

        //區域jacobian公式中的"Im(x_k, w_m) x delta_t"
        warped_image_delta_t(v_iter) = - eventWarped.time_delta(e_valid_v_iter)
                                * warped_event_image.at<float>(coord_y, coord_x);
        if(rotation_estimation){
            Ix_interp_map(v_iter) = Ix_sum.at<float>(coord_y, coord_x);
            Iy_interp_map(v_iter) = Iy_sum.at<float>(coord_y, coord_x);

            //全域jacobian公式中的"(I_L + I_G)"
            warped_map_delta_t(v_iter) = - truncated_sum.at<float>(coord_y, coord_x);
        }
    }
    Ix_interp *= camParam.fx;
    Iy_interp *= camParam.fy;
    //看起來有點亂，為jacobian公式中的中間兩項相乘
    xp_zp = xp.array() / zp.array();
    yp_zp = yp.array() / zp.array();
    Eigen::ArrayXf xx_zz = xp_zp.array() * xp_zp.array();
    Eigen::ArrayXf yy_zz = yp_zp.array() * yp_zp.array();
    Eigen::ArrayXf xy_zz = xp_zp.array() * yp_zp.array();
    Jacobian.col(0) = -(Ix_interp.array() * xy_zz)              //cv::Mat 需改寫////////////////////
                    - (Iy_interp.array() * (1 + yy_zz));
    Jacobian.col(1) = (Ix_interp.array() * (1 + xx_zz))
                    + (Iy_interp.array() * xy_zz);
    Jacobian.col(2) = -Ix_interp.array() * yp_zp.array() + Iy_interp.array() * xp_zp.array();

    if(rotation_estimation){
        Ix_interp_map *= camParam.fx;
        Iy_interp_map *= camParam.fy;

        Jacobian_map.col(0) = -(Ix_interp_map.array() * xy_zz)
                            - (Iy_interp_map.array() * (1 + yy_zz));
        Jacobian_map.col(1) = (Ix_interp_map.array() * (1 + xx_zz))
                            + (Iy_interp_map.array() * xy_zz);
        Jacobian_map.col(2) = -Ix_interp_map.array() * yp_zp.array() + Iy_interp_map.array() * xp_zp.array();
    }
}

// warping function
void System::GetWarpedEventPoint(const EventBundle &eventIn, EventBundle &eventOut,
    const Eigen::Vector3f &temp_ang_vel, const bool &is_map_warping,
    const Eigen::Vector3f &temp_ang_pos ){     // is_map_warping = false,   temp_ang_pos = Eigen::Vector3f::Zero()
    if(eventIn.size == 0){
        std::cout << "EventIn size is zero" << std::endl;
        return;
    }
    float ang_vel_norm = temp_ang_vel.norm();
    float ang_pos_norm = temp_ang_pos.norm();
    Eigen::MatrixXf x_ang_vel_hat, x_ang_vel_hat_square;
    x_ang_vel_hat.resize(eventIn.size, 3);
    x_ang_vel_hat_square.resize(eventIn.size, 3);

    //rodrigues formula first-order term
    x_ang_vel_hat.col(0) = - temp_ang_vel(2) * eventIn.coord_3d.col(1) + temp_ang_vel(1) * eventIn.coord_3d.col(2);
    x_ang_vel_hat.col(1) = + temp_ang_vel(2) * eventIn.coord_3d.col(0) - temp_ang_vel(0) * eventIn.coord_3d.col(2);
    x_ang_vel_hat.col(2) = - temp_ang_vel(1) * eventIn.coord_3d.col(0) + temp_ang_vel(0) * eventIn.coord_3d.col(1);
    //rodrigues formula second-order term
    x_ang_vel_hat_square.col(0) = - temp_ang_vel(2) * x_ang_vel_hat.col(1) + temp_ang_vel(1) * x_ang_vel_hat.col(2);
    x_ang_vel_hat_square.col(1) = + temp_ang_vel(2) * x_ang_vel_hat.col(0) - temp_ang_vel(0) * x_ang_vel_hat.col(2);
    x_ang_vel_hat_square.col(2) = - temp_ang_vel(1) * x_ang_vel_hat.col(0) + temp_ang_vel(0) * x_ang_vel_hat.col(1);

    // Element-wise
    // points warping via second-order approximation with Rodrigue's formular
    //先做區域對齊，再轉到全域對齊
    if(is_map_warping){
        if(ang_vel_norm < 1e-8){
            eventOut.coord_3d = eventIn.coord_3d;
        }
        else{
            //rodrigues formula second-order approximation
            auto delta_t = eventIn.time_delta_reverse.array();
            eventOut.coord_3d = eventIn.coord_3d
                            + Eigen::MatrixXf(x_ang_vel_hat.array().colwise()
                            * delta_t.array()
                            + x_ang_vel_hat_square.array().colwise()
                            * (0.5f * delta_t.array().square()) );
        }
    }
    else{
        if(ang_vel_norm < 1e-8){
            eventOut.coord_3d = eventIn.coord_3d;
        }
        else{
            auto delta_t = eventIn.time_delta.array();
            eventOut.coord_3d = eventIn.coord_3d
                            + Eigen::MatrixXf( x_ang_vel_hat.array().colwise()
                            * delta_t.array()
                            + x_ang_vel_hat_square.array().colwise()
                            * (0.5f * delta_t.array().square()) );
        }
    }
    //轉回參考位置，local alignment to global alignment
    if(ang_pos_norm > 1e-8){
        eventOut.coord_3d = eventOut.coord_3d * SO3(temp_ang_pos).transpose();
    }
}

std::vector<uint16_t> System::GetValidIndexFromEvent(const EventBundle & event){
    std::vector<uint16_t> valid;
    uint16_t valid_counter = 0;
    for (auto e_iter = event.isInner.begin(); e_iter != event.isInner.end(); e_iter++){
        if(*e_iter){
            valid.push_back(valid_counter);
        }
        valid_counter++;
    }
    return valid;
}

void System::GetWarpedEventImage(const cv::Vec3f &temp_ang_vel, const PlotOption &option){      // option = UNSIGNED_EVENT_IMAGE
    GetWarpedEventPoint(eventUndistorted, eventWarped, temp_ang_vel);
    eventWarped.Projection(K);
    eventWarped.SetXY();
    eventWarped.DiscriminateInner(width - 1, height - 1);
    GetEventImage(eventWarped, option).convertTo(warped_event_image, CV_32FC1);
}

void System::GetWarpedEventImage(const Eigen::Vector3f &temp_ang_vel, const Eigen::Vector3f &temp_ang_pos){
    GetWarpedEventPoint(eventUndistorted, eventWarped, temp_ang_vel, false, temp_ang_pos);
    eventWarped.Projection(K);
    eventWarped.SetXY();
    eventWarped.DiscriminateInner(width - 1, height - 1);
    if(is_polarity_on){
        GetEventImage(eventWarped, SIGNED_EVENT_IMAGE).convertTo(warped_event_image, CV_32FC1);
    }
    else{
        GetEventImage(eventWarped, UNSIGNED_EVENT_IMAGE).convertTo(warped_event_image, CV_32FC1);
    }
}

void System::GetWarpedEventMap(const Eigen::Vector3f &temp_ang_pos){
    if( eventMap.size == 0 ){
        return;
    }
    GetWarpedEventPoint(eventMap, eventWarpedMap, Eigen::Vector3f::Zero(), true, -temp_ang_pos);
    eventWarpedMap.Projection(K);
    eventWarpedMap.SetXY();
    eventWarpedMap.DiscriminateInner(width - 1, height - 1, map_sampling_rate);
    GetEventImage(eventWarpedMap, UNSIGNED_EVENT_IMAGE, false).convertTo(warped_map_image, CV_32FC1);
}

cv::Mat System::GetMapImage(const PlotOption &option){           // option = UNSIGNED_EVENT_IMAGE
    cv::Mat map_image = cv::Mat(height_map, width_map, CV_16UC3);       //累積的映射結果圖
    map_image = cv::Scalar(0, 0, 0);
    cv::Mat map_image_temp = cv::Mat(height_map, width_map, CV_16UC3);
    EventBundle temp;
    for (uint16_t m_iter = map_iter_begin; m_iter != eventMapPoints.size(); m_iter++){
        EventBundle & curEvent = eventMapPoints.at(m_iter);
        curEvent.size = curEvent.coord_3d.rows();
        if(curEvent.size == 0 || SO3add(curEvent.angular_position, -angular_position).norm() > mapping_interval){
            // map_iter_begin = m_iter + 1;
            continue;
        }
        temp.Copy(curEvent);
        GetWarpedEventPoint(curEvent, temp, Eigen::Vector3f::Zero(), true, -angular_position);
        temp.Projection(K_map);
        temp.SetXY();
        temp.DiscriminateInner(width_map - 1, height_map - 1, map_sampling_rate);
        GetEventImage(temp, option, true).convertTo(map_image_temp, CV_16UC3);  //其實SIGNED_EVENT_IMAGE_COLOR已經是CV_16UC3
        map_image += map_image_temp;
    }
    cv::Mat result;
    map_image.convertTo(result, CV_32FC3);
    return result;
}

cv::Mat System::GetMapImageNew(const PlotOption &option){               // option = UNSIGNED_EVENT_IMAGE
    cv::Mat result = cv::Mat(height_map, width_map, CV_32FC3);          //映射結果圖
    result = cv::Scalar(0, 0, 0);
    EventBundle eventMapTemp;
    eventMapTemp.Copy(eventMap);
    if( eventMap.size == 0 ){
        return result;
    }
    GetWarpedEventPoint(eventMap, eventMapTemp, Eigen::Vector3f::Zero(), true, -angular_position);
    eventMapTemp.Projection(K_map);
    eventMapTemp.SetXY();
    eventMapTemp.DiscriminateInner(width_map - 1, height_map - 1, map_sampling_rate);
    GetEventImage(eventMapTemp, option, true).convertTo(result, CV_32FC3);
    return result;
}
*/