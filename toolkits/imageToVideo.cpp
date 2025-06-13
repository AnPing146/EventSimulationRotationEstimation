#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "numerics.h"

using namespace std;
//using namespace cv;

int intCount(int i);
int main(int argc, char **argv)
{
    // ./toolkits/imageToVideo /home/chengwei/program/DSEC/interlaken_00_c/interlaken_00_c_images_rectified_left dsec interlaken_00_c
    // ./toolkits/imageToVideo /home/chengwei/program/DSEC/thun_00_a/thun_00_a_images_rectified_left dsec thun_00_a

    // ./toolkits/imageToVideo /home/chengwei/code/thun_00_a_images_rectified_left dsec thun_00_a
    // ./toolkits/imageToVideo ../experiments/images/images_boxes ecds boxes_rotation
    // ./toolkits/imageToVideo ../experiments/images/images_poster ecds poster_rotation
    // ./toolkits/imageToVideo ../experiments/testframe dsec testframe
    
    // ./toolkits/imageToVideo ../images_0_226 ecds shapes_rotation_0_226
    // ./toolkits/imageToVideo ../images_227_453 ecds shapes_rotation_227_453


    if(argc!=4){
        cout << "***\nHints:\n"
        << setw(6) << "" << "argrment[1]: image path\n"
        << setw(6) << "" << "argrment[2]: dataset(\"dsec\" or \"ecds\")\n"
        << setw(6) << "" << "argument[3]: output video name\n***\n";

        return -1;
    }

    vector<cv::Mat> images;
    int fps;
    string str_zero, temp_name;

    string imagePath(argv[1]);
    if(argv[2] == string("dsec")){
        fps = 20;
        temp_name = string("");
        str_zero = string("/000000");
  
    }else if(argv[2] == string("ecds")){
        fps = 24;
        temp_name = string("/frame_0");
        str_zero = string("0000000");
    }else{
        cout<<"Invalid dataset! Check dataset name!\n";

        return -1;
    }

    string fullPath = imagePath + temp_name + str_zero + ".png";
    cout<<fullPath<<"\n";

    cv::Mat img = cv::imread(fullPath);
    if (img.empty())
    {
        cout << "Fail to open image! Check file path!" << endl;
        return -1;
    }
    images.emplace_back(img);

    int i = 0;
    while (++i)
    {
        int length = intCount(i);
        str_zero.replace(7 - length, length, to_string(i));
        string fullPath = imagePath + temp_name + str_zero + ".png";
        cout<<fullPath<<endl;
        img = cv::imread(fullPath);
        
        if (img.empty()) break;
        images.emplace_back(gammaCorrect(img, 0.5));   //gammaCorrect(img, 0.5) img
    }
    cout << "read " << i-1 << " images finish!" << endl;

    /*
    for (int i = 0; i < images.size(); i++)
    {
        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        cv::imshow("result", images[i]);
        cout << "image: " << i << endl;

        cv::waitKey(50);
    }    
    */

    cv::VideoWriter video_capture;
    auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    video_capture.open(string(argv[3]) + "_video.mp4", fourcc, fps/*fps*/,
                       cv::Size(images[0].cols, images[0].rows));
    if (video_capture.isOpened())
    {
        for (int i = 0; i < images.size(); i++)
        {
            video_capture << images[i];
        }
    }
    else{
        cout<<"can't open video cpautre for video write\n";
    }
    video_capture.release();
    cout << "video saved!" << endl;

    return 0;
}

int intCount(int i)
{
    int length = 1;
    while (i /= 10)
        length++;

    return length;
}