cmake_minimum_required(VERSION 3.15)

find_package(OpenCV REQUIRED)

#add_subdirectory(source)

add_executable(rectifyImageOrVideo rectifyImageOrVideo.cpp)
target_link_libraries(rectifyImageOrVideo ${OpenCV_LIBS} eventSimulator)

add_executable(imageToVideo imageToVideo.cpp)
target_link_libraries(imageToVideo ${OpenCV_LIBS} numerics)

add_executable(videoToImage videoToImage.cpp)
target_link_libraries(videoToImage ${OpenCV_LIBS})

add_executable(videoCompare videoCompare.cpp)
target_link_libraries(videoCompare ${OpenCV_LIBS})