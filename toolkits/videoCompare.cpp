#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

int main(int argc, char **argv)
{
    /*
    ./toolkits/videoCompare \
    ./thun_00_a_event_rectify.mp4 \
    ./thun_00_a_video_rectify.mp4 \
    ./SparseInterpolatedEventSimulator_LKOpticalFlowCalculator_f_p_n_pi_pn_10_10_10_video.mp4 \
    ./SparseInterpolatedEventSimulator_LKOpticalFlowCalculator_f_p_n_pi_pn_10_10_10_flow.mp4 \
    compareVideo_lucaskanade.mp4

    ./toolkits/videoCompare \
    ./thun_00_a_event_rectify.mp4 \
    ./thun_00_a_video_rectify.mp4 \
    ./DenseInterpolatedEventSimulator_FarnebackFlowCalculator_f_p_n_pi_pn_10_2_2_video.mp4 \
    ./DenseInterpolatedEventSimulator_FarnebackFlowCalculator_f_p_n_pi_pn_10_2_2_flow.mp4 \
    compareVideo_farneback.mp4
    
    ./toolkits/videoCompare \
    ./thun_00_a_event_rectify.mp4 \
    ./thun_00_a_video_rectify.mp4 \
    ./DenseInterpolatedEventSimulator_DISOpticalFlowCalculator_2_f_p_n_pi_pn_10_2_2_video.mp4 \
    ./DenseInterpolatedEventSimulator_DISOpticalFlowCalculator_2_f_p_n_pi_pn_10_2_2_flow.mp4 \
    compareVideo_DenseInverseSearch.mp4

    ./toolkits/videoCompare \
    ./thun_00_a_event_rectify.mp4 \
    ./thun_00_a_video_rectify.mp4 \
    ./DifferenceInterpolatedEventSimulator_LKOpticalFlowCalculator_f_p_n_pi_pn_10_20_20_30_30_video.mp4 \
    ./DifferenceInterpolatedEventSimulator_LKOpticalFlowCalculator_f_p_n_pi_pn_10_20_20_30_30_flow.mp4 \
    compareVideo_DifInterp_LK.mp4
    */
    
    if(argc!=6){
        cout << "***\nHints:\n"
        << setw(6) << "" << "argrment[1]: video0 path(cam0:事件相機)\n"
        << setw(6) << "" << "argrment[2]: video2 path(cam1一般相機)\n"
        << setw(6) << "" << "argrment[3]: video3 path(模擬事件)\n"
        << setw(6) << "" << "argrment[4]: video4 path(光流)\n"
        << setw(6) << "" << "argument[5]: output file name\n***\n";

        return -1;
    }

    // video read
    cv::VideoCapture cap1(argv[1]), cap2(argv[2]), cap3(argv[3]), cap4(argv[4]);
    if(!cap1.isOpened()){
        cout<<"Can't open video[1]! Check video[1] path.\n";
        return -1;
    }
    if(!cap2.isOpened()){
        cout<<"Can't open video[2]! Check video[2] path.\n";
        return -1;
    }
    if(!cap3.isOpened()){
        cout<<"Can't open video[3]! Check video[3] path.\n";
        return -1;
    }
    if(!cap4.isOpened()){
        cout<<"Can't open video[4]! Check video[4] path.\n";
        return -1;
    }

    // image concatenation, then write video
    cv::Mat images[4];
    cv::Mat temp, result;
    int rows, cols, length, count = 0;

    cap1>>images[0];        //cam0, event
    cap2>>images[1];        //cam1, frame
    cap3>>images[2];        //event simuation
    cap4>>images[3];        //optical flow
    rows = images[2].rows;
    cols = images[2].cols;
    length = cap2.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter video_writer;
    auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    video_writer.open(string(argv[5]), fourcc, 20 /*fps*/,
                       cv::Size(cols*2, rows*2));
    if (video_writer.isOpened())
    {
        cv::hconcat(images[0], images[1], result);
        cv::hconcat(images[2], images[3], temp);
        cv::vconcat(result, temp, result);
        video_writer << result;
        count++;

        //cv::namedWindow("result");
        //cv::imshow("result", temp);
        //cv::waitKey(1);

        cout<<"cap1 frame count: "<<cap1.get(cv::CAP_PROP_FRAME_COUNT)<<"\n";
        cout<<"cap2 frame count: "<<cap2.get(cv::CAP_PROP_FRAME_COUNT)<<"\n";
        cout<<"cap3 frame count: "<<cap3.get(cv::CAP_PROP_FRAME_COUNT)<<"\n";
        cout<<"cap4 frame count: "<<cap4.get(cv::CAP_PROP_FRAME_COUNT)<<"\n\n";

        for (int i = 0; i < length-1; i++, count++)
        {
            cap1>>images[0];        
            cap2>>images[1];   
            cap3>>images[2];     
            cap4>>images[3];  

            if(images[0].empty()){
                cout<<"images[0]: "<< count<<" empty!\n";   //event
                break;
            }else if(images[1].empty()){
                cout<<"images[1]: "<< count<<"  empty!\n";  //frame
                break;
            }else if(images[2].empty()){
                cout<<"images[2]: "<< count<<"  empty!\n";  //event simulation 
                break;
            }else if(images[3].empty()){
                cout<<"images[3]: "<< count<<"  empty!\n";  //flow
                break;
            }

            cout<<"count: "<< count<<"\n";
            cout<<"images[0].size(): "<<images[0].size()<<"\n";
            cout<<"images[1].size(): "<<images[1].size()<<"\n";
            cout<<"images[2].size(): "<<images[2].size()<<"\n";
            cout<<"images[3].size(): "<<images[3].size()<<"\n";    

            cv::hconcat(images[0], images[1], result);
            cv::hconcat(images[2], images[3], temp);
            cv::vconcat(result, temp, result);

            cv::imshow("result", result);
            cv::waitKey(1);

            video_writer << result;
        }
    }
    else{
        cout<<"can't open video cpautre for video write\n";
    }
    video_writer.release();
    cout << count<< "frames, video saved!" << endl;

    return 0;
}
