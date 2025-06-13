cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

project(eventSimulatorTest LANGUAGES CXX VERSION 0.1.0)

###for win environment###
set(OpenCV_DIR "C:/opencv/build_mingw/install") # Adjust this path to your OpenCV build directory
find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIR})

###for ubuntu environment###
##set(OpenCV_DIR "/home/chengwei/opencv/build")
##set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/header/eigen-3.4.0)
#find_package(OpenCV REQUIRED)

make_directory("images/events")
make_directory("images/frames")
make_directory("rectifyImage/events")
make_directory("rectifyImage/frames")

make_directory("../experiments/videos")
make_directory("../experiments/images")
make_directory("../experiments/events")

add_subdirectory(source)
add_subdirectory(toolkits)
add_subdirectory(experiments)

add_executable(eventFromStream eventFromStream.cpp)
target_link_libraries(eventFromStream ${OpenCV_LIBS} eventSimulator)

add_executable(eventFromVideo eventFromVideo.cpp)
target_link_libraries(eventFromVideo ${OpenCV_LIBS} eventSimulator)

add_executable(eventFromDataset eventFromDataset.cpp)
target_link_libraries(eventFromDataset ${OpenCV_LIBS} eventSimulator)



