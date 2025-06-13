#ifndef SYSTEM_H
#define SYSTEM_H

// #include <Eigen/Core>

#include <string>
#include <vector>
#include <fstream>
//#include <deque>

#include "Event.h"

enum PlotOption
{
    DVS_RENDERER,
    SIGNED_EVENT_IMAGE,
    SIGNED_EVENT_IMAGE_COLOR,
    UNSIGNED_EVENT_IMAGE,
    GRAY_EVENT_IMAGE
};

class System
{
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW             //eigen巨集，確保eigen資料結構記憶體對齊
    System(const std::string &strSettingsFile);

    ~System();

    // Data read
    void PushImageData(ImageData imageData);
    void PushImuData(ImuData imuData);
    // void PushEventData(EventData eventData);

    /**
     * @brief 從事件向量中取出獨立事件並取樣為事件包，並運行Run()或RunUptoIdx()
     */
    void BindEvent(std::vector<Event> &eventData);

    /**
     * @brief 取得歷史時間戳
     */
    std::vector<float> GetRecordTimestamp() { return historical_timestamp; }

    /**
     * @brief 取得歷史角速度
     */
    std::vector<cv::Vec3d> GetRecordVelocity() { return historical_angular_velocity; }

    /**
     * @brief 取得歷史角位置
     */
    std::vector<cv::Vec3d> GetRecordPosition() { return historical_angular_position; }    

private:
    /**
     * @brief run the global alignment system
     */
    void Run();

    /**
     * @brief 事件包累積到RunUptoIdx數量後才運行旋轉估測系統. 
     * view_index次後才顯示CM圖
     * 注意: run_index要大於view_index才會顯示CM結果
     */
    void RunUptoIdx();

    /**
     * @brief 從eventMapPoints中挑選符合mapping_interval的事件到eventMap與eventWarpedMap，避免姿態差異太大，導致對齊失敗
     * eventWarpedMap缺少角度資訊，但後續GetWarpedEventPoint()時會補上角度數據
     * 
     * 注意:如果角度移動一直很小，則eventMap會一直累積，系統會越來越慢! 直到mapping_interval的角度差異才會清除舊資料
     */
    void BindMap();

    /**
     * @brief 計算完一次後會清除舊資料. 清除事件包、校正的事件、扭轉的事件，但會保留全域對齊事件
     */
    void ClearEvent();

    /**
     * @brief 事件稀疏校正
     */
    void UndistortEvent();

    /**
     * @brief 影像校正
     */
    void UndistortImage();

    /**
     * @brief 產生影像校正網格
     */
    void GenerateMesh();

    /**
     * @brief globally alignment contrast maximization algorithm
     * Gradient = del_C(w) = sigma_k 2 * I(x_k) * partial_I(x_k) / partial_w = sigma_k 2 * I(x_k) * Jacobian
     */
    void EstimateMotion();

    /**
     * @brief 計算扭轉影像，輸出到warped_event_image
     */
    void GetWarpedEventImage(const cv::Vec3d &temp_ang_vel, const PlotOption &option = UNSIGNED_EVENT_IMAGE);

    /**
     * @brief 計算扭轉並對齊影像，由is_polarity_on變數控制是否計算極性，輸出到warped_event_image
     */
    void GetWarpedEventImage(const cv::Vec3d &temp_ang_vel, const cv::Vec3d &temp_ang_pos);

    /**
     * @brief 由全域對齊扭轉回當前位置，再投影到目前影像平面，並取得warped_map_image
     */
    void GetWarpedEventMap(const cv::Vec3d &temp_ang_pos);

    /**
     * @brief warping function. 求出eventOut.coord_3d
     * @param eventIn
     * @param eventOut
     * @param temp_ang_vel temporal alignment by angular velocity
     * @param is_map_warping False indicate using backward warping(normal warping). On the other
     * hand, True indicate forward warping(reverse warping).
     * @param temp_ang_pos global alignment by angular position
     * 如果是反向，global to local則輸入"-temp_ang_pos"
     */
    void GetWarpedEventPoint(const EventBundle &eventIn,
                             EventBundle &eventOut,
                             const cv::Vec3d &temp_ang_vel,
                             const bool &is_map_warping = false,
                             const cv::Vec3d &temp_ang_pos = cv::Vec3d::all(0.0));

    // void DeriveErrAnalytic(const Eigen::Vector3f &temp_ang_vel);

    /**
     * @brief 計算影像梯度、jacobian矩陣，注意全域對齊時包含取出視窗內的有效像素!
     * Jacobian = partial_I(x_k) / partial_w = del_I(x_k) * partial_x_k / partial_w
     */
    void DeriveErrAnalytic(const cv::Vec3d &temp_ang_vel, const cv::Vec3d &temp_ang_pos);

    // void DeriveErrAnalyticMap(const Eigen::Vector3f &temp_ang_pos);

    /**
     * @brief 取出isInner的事件索引值
     */
    std::vector<uint16_t> GetValidIndexFromEvent(const EventBundle &event);
    // void JacobianTrim();

    /**
     * @brief 前一刻事件包的全域對齊，儲存前一刻的速度與目前對齊的位置，再計算初始化角位置
     */
    void ComputeAngularPosition();

    // void UpdateAngularVelocity();
    // void UpdateAngularPosition();
    // void Mapping();

    /**
     * @brief 映射"累積"對齊圖
     */
    cv::Mat GetMapImage(const PlotOption &option = SIGNED_EVENT_IMAGE);

    /**
     * @brief 映射對齊圖
     */
    cv::Mat GetMapImageNew(const PlotOption &option = SIGNED_EVENT_IMAGE);

    void Visualize();
    void Renderer();

    /**
     * @brief 依option條件產生事件包x-y資料的影像
     */
    cv::Mat GetEventImage(const EventBundle &e, const PlotOption &option, const bool &is_large_size = 0);

    uint32_t bindCnt;
    /// variables
    // data stack
    std::vector<ImuData> vec_imu_data;
    std::vector<ImageData> vec_image_data;
    // std::vector<EventData> vec_event_data;          // 原始事件向量(EventData包含ros-msg)
    std::vector<EventBundle> vec_event_bundle_data; // 原始事件包向量( RunUptoIdx()批次處理用 )

    // event points
    EventBundle eventBundle;
    EventBundle eventUndistorted;            // 應更名 eventBundleUndist
    EventBundle eventUndistortedPrev;        // 應更名 eventBundleUndistPrev
    EventBundle eventWarped;                 // copy by eventUndistorted
    EventBundle eventAlignedPoints;          // copy by eventUndistortedPrev
    EventBundle eventMap;                    // 全域對齊事件包 Globally Aligned Events
    EventBundle eventWarpedMap;              // 挑選過的eventMap，用於反向投影
    std::vector<EventBundle> eventMapPoints; // 全域對齊事件包向量

    // Image
    ImageData current_image;
    ImageData undistorted_image;
    cv::Mat undistorted_render;
    cv::Mat map_render;
    cv::Mat undist_mesh_x;
    cv::Mat undist_mesh_y;

    // time
    float next_process_time;
    float delta_time;
    float event_delta_time;
    float last_event_time;                   // 前一次事件包的第一刻時間
    float estimation_time_interval;          // 更新每次的時間間隔
    float estimation_time_interval_prev;

    int max_event_num;

    // counter
    uint16_t map_iter_begin;                 // 紀錄已挑選過的全域對齊事件包

    // size_t event_data_iter; // vector_event_data的索引，將事件放入eventBundle
    // size_t image_data_iter;
    // size_t imu_data_iter;

    // samplier
    int sampling_rate;
    int map_sampling_rate;
    uint32_t sliding_window_size;                 // 滑動視窗大小

    // camera parameters
    cv::Matx33d K;     // cameraMatrix
    cv::Matx33d K_map; // 映射圖用的相機參數矩陣(可調整映射比例)
    CameraParam camParam;
    // cv::Matx33f cameraMatrix;
    cv::Matx14d distCoeffs;
    int width, height;
    int width_map, height_map;
    float map_scale;

    // camera pose
    std::vector<float> estimatior_time;

    std::vector<cv::Vec3d> vec_angular_velocity;    //
    std::vector<cv::Vec3d> vec_angular_position;    //

    cv::Vec3d angular_velocity; // 會用到角度函式 所以用Vec不用Mat
    cv::Vec3d angular_velocity_prev;
    cv::Vec3d angular_position;
    cv::Vec3d angular_position_init; // initial position for optimization
    cv::Vec3d angular_position_prev;
    cv::Vec3d angular_position_pprev;
    cv::Vec3d angular_position_compensator;

    cv::Vec3d update_angular_velocity;
    cv::Vec3d update_angular_position;

    // visualize
    float plot_denom;
    float denominator;

    /// Estimator variables
    // optimization parameters
    double nu_event;  // exponential average of squares of gradients
    double eta_event; // step size | optimization rate
    double rho_event; // smoothing factor | the degree of weigthing decrease in geometric moving average

    double nu_map;
    double eta_map;
    double rho_map;

    int optimizer_max_iter;
    bool is_polarity_on;

    cv::Mat warped_event_image; // 扭轉(並對齊)影像
    cv::Mat warped_event_image_grabbed;
    cv::Mat warped_map_image; // 全域對齊投影回相機影像平面的影像 CV_32FC1

    // derive error analytic
    // cv::Mat grad_x_kernel;    //沒用到
    // cv::Mat grad_y_kernel;

    std::vector<float> xp_zp; // 論文中的x_k_hat, 即x_prime/z_prime
    std::vector<float> yp_zp;

    std::vector<float> xx_zz; // jacobian矩陣係數
    std::vector<float> yy_zz;
    std::vector<float> xy_zz;

    std::vector<float> xp;
    std::vector<float> yp;
    std::vector<float> zp;

    cv::Mat blur_image;

    cv::Mat Ix_sum;
    cv::Mat Iy_sum;

    cv::Mat Ix; // 區域對齊影像的梯度
    cv::Mat Iy;
    std::vector<float> Ix_interp; // 插值法取出有效值位置的梯度，即"partial_I / partial_X"  Eigen::VectorXf
    std::vector<float> Iy_interp; // Eigen::VectorXf

    // Eigen::VectorXf xp_map;     //沒用到
    // Eigen::VectorXf yp_map;     //Eigen::VectorXf
    // Eigen::VectorXf zp_map;

    // cv::Mat blur_image_map;     //沒用到
    cv::Mat truncated_map;   // global alignment event image
    cv::Mat truncated_event; // local alignemnt event image
    cv::Mat truncated_sum;   // global-local alignment sum

    cv::Mat Ix_map; // 全域對齊影像的梯度
    cv::Mat Iy_map;
    std::vector<float> Ix_interp_map; // 插值法取出"全域"有效值位置的梯度，即"partial_I / partial_X"   Eigen::VectorXf
    std::vector<float> Iy_interp_map;

    std::vector<float> warped_image_delta_t; // 區域jacobian公式中的"Im(x_k, w_m) x delta_t"        Eigen::VectorXf
    std::vector<float> warped_map_delta_t;   // 全域jacobian公式中的"(I_L + I_G)"

    cv::Mat Jacobian; // n-by-3
    cv::Mat Jacobian_map;

    cv::Matx31d Gradient; // Eigen::Vector3f
    cv::Matx31d Gradient_map;

    int event_image_threshold;
    bool rotation_estimation; // 是否使用global alignment algorithm
    int run_index;
    double mapping_interval; // 會與 double cv::norm() 比較
    int optimization_view_index;
    bool optimization_view_flag;

    // log data
    std::vector<cv::Vec3d> historical_angular_velocity;
    std::vector<cv::Vec3d> historical_angular_position;
    std::vector<float> historical_timestamp;

    std::string filePath;
    std::ofstream writeFile;
};

#endif // SYSTEM_H