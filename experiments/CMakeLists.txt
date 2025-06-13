cmake_minimum_required(VERSION 3.15)

find_package(OpenCV REQUIRED)

add_executable(DISFlow_param DISFlow_param.cpp)
target_link_libraries(DISFlow_param ${OpenCV_LIBS} flowColor)

add_executable(eventSim_param eventSim_param.cpp)
target_link_libraries(eventSim_param ${OpenCV_LIBS} eventSimulator)

add_executable(eventSim_check eventSim_check.cpp)
target_link_libraries(eventSim_check ${OpenCV_LIBS} eventSimulator)

add_executable(eventSim_EPS eventSim_EPS.cpp)
target_link_libraries(eventSim_EPS ${OpenCV_LIBS} eventSimulator)