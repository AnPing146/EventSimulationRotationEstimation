#include <iostream>
#include <opencv2/opencv.hpp>

#include "EventSimulators.h"
#include "Player.h"

using namespace std;
using namespace cv;
int main(int argc, char **argv)
{
    
    int x_res = 640;
    int y_res = 480;
    int wait_time_ms = 0;
    int C_pos = 20, C_neg = 20, num_inter_frames = 10;
    int div_factor = 10;
    int C_offset = 10;
    int DISQuality = 0;

    //std::shared_ptr<DenseOpticalFlowCalculator> ofc_fb = std::make_shared<FarnebackFlowCalculator>();
    std::shared_ptr<DenseOpticalFlowCalculator> ofc_dis_low = std::make_shared<DISOpticalFlowCalculator>(DISQuality);
    //std::shared_ptr<SparseOpticalFlowCalculator> ofc_lk = std::make_shared<LKOpticalFlowCalculator>();

    //std::shared_ptr<EventSimulator> bdf = std::make_shared<BasicDifferenceEventSimulator>(C_pos, C_neg);
    //std::shared_ptr<EventSimulator> dsmes = std::make_shared<DenseInterpolatedEventSimulator>(ofc_fb, num_inter_frames, C_pos / div_factor, C_neg / div_factor);
    std::shared_ptr<EventSimulator> dsmes = std::make_shared<DenseInterpolatedEventSimulator>(ofc_dis_low, num_inter_frames, C_pos / div_factor, C_neg / div_factor);    
    //std::shared_ptr<EventSimulator> ssmes = std::make_shared<SparseInterpolatedEventSimulator>(ofc_lk, num_inter_frames, C_pos / 2, C_neg / 2);
    ////std::shared_ptr<EventSimulator> sdies = std::make_shared<SparseInterpolatedEventSimulator>(ofc_lk, num_inter_frames, C_pos, C_neg, C_pos + C_offset, C_neg + C_offset);

    // std::string video_path =  "../res/videos/car.mp4";
    // OpenCVPlayer cv_player = OpenCVPlayer(dsmes, wait_time_ms);
    // cv_player.play(video_path);

    VideoStreamer streamer = VideoStreamer(dsmes);
    streamer.simulateFromStream(0);

    return 0;
    
}
