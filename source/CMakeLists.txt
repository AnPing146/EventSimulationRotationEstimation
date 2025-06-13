cmake_minimum_required(VERSION 3.15)

find_package(OpenCV REQUIRED)

#set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/header/eigen-3.4.0)

add_library(eventSimulator Player.cpp 
                            Event.cpp 
                            EventSimulators.cpp 
                            OpticalFlowCalculator.cpp 
                            System.cpp 
                            Estimator.cpp 
                            Undistortion.cpp
                            numerics.cpp)
target_include_directories(eventSimulator PUBLIC ${CMAKE_SOURCE_DIR}/header)    #${EIGEN_INCLUDE_DIR}
target_link_libraries(eventSimulator ${OpenCV_LIBS})

add_library(numerics numerics.cpp)
target_include_directories(numerics PUBLIC ${CMAKE_SOURCE_DIR}/header)    #${EIGEN_INCLUDE_DIR}
target_link_libraries(numerics ${OpenCV_LIBS})

add_library(flowColor flowColor.cpp)
target_include_directories(flowColor PUBLIC ${CMAKE_SOURCE_DIR}/header)    
target_link_libraries(flowColor ${OpenCV_LIBS})

