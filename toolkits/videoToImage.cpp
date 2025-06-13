#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
int intCount(int i);
int main(int argc, char **argv){
    // ./toolkits/videoToImage ./thun_00_a_event.mp4
    // ./toolkits/videoToImage ./thun_00_a_video.mp4
    if(argc!=2){
        cout << "***\nHints:\n" 
        << std::setw(6) << "" << "argrment[1]: video path\n***\n";

        return -1;
    }
    
    string outputCount("000000");
    cv::Mat frame;

    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened()){
        cout<<"Can't open video! Check video path.\n";
        return -1;
    }

    // /home/chengwei/program/eventSimulatorTest/build
    int frameSize = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    for(int i=0; i<frameSize; i++){
        int length = intCount(i);
        outputCount.replace(6 - length, length, to_string(i));
        string fullPath("./images/" + outputCount + ".png");

        cap>>frame;
        cv::imwrite(fullPath, frame);
    }
    cout<<"Finished "<<cap.get(cv::CAP_PROP_FRAME_COUNT)<< " frame !!\n";
    cap.release();

    return 0;
}
int intCount(int i)
{
    int length = 1;
    while (i /= 10)
        length++;

    return length;
}