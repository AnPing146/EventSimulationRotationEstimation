#include <iostream>
#include <opencv2/opencv.hpp>

#include "EventSimulators.h"
#include "Player.h"

using namespace std;
using namespace cv;
int main(int argc, char **argv)
{
    uint16_t bunchSize = 1000;

    std::string event_path = "../experiments/events/events_shapes.txt";
    std::string config_path = "../config/config_ECDS.yaml";
    std::cout << "Using configure file: " << config_path << "\n";

    std::shared_ptr<System> sys = std::make_shared<System>(config_path);

    EventDataPlayer eventPlayer(cv::Rect(0,0,0,0));
    eventPlayer.setSystem(sys);

    if (eventPlayer.readEventFile(event_path))
    {
        eventPlayer.publishData(bunchSize);
    }

    return 0;
}
