cmake_minimum_required(VERSION 3.8)
project(rm_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 平台配置
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)
set(FORCE_OPENVINO OFF)
set(OPENVINO_2022 ON)

# find ament_cmake
find_package(ament_cmake_auto REQUIRED)
find_package(rm_interfaces REQUIRED)
find_package(rm_utils REQUIRED)
ament_auto_find_build_dependencies()

# OpenCV
find_package(OpenCV 4 REQUIRED)
set(link ${OpenCV_LIBS} ${link})

# Thread
find_package(Threads REQUIRED)
set(link Threads::Threads ${link})

# # try to find CUDA
# find_package(CUDA OPTIONAL_COMPONENTS)
# if(${CUDA_FOUND} AND NOT FORCE_OPENVINO)
#     add_compile_options(-DTRT)
#     message(STATUS "TensorRT Mode")
#     set(TRT ON)
# else()
#     message(STATUS "OpenVINO Mode")
    set(TRT OFF)
# endif()

if(TRT)
    set(include ${CUDA_INCLUDE_DIRS} ${include})
    set(link ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers nvonnxparser cuda z ${link})
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        message("-- x86_64 tar_file TensorRT")
        set(include "/home/cx/3rdParty/TensorRT-8.6.1.6/include" ${include})
    endif()
else()
    # OpenVINO
    if(OPENVINO_2022)
        find_package(OpenVINO REQUIRED)
        set(link openvino::runtime ${link})
    else()
        set(InferenceEngine_DIR "/opt/intel/openvino/runtime/cmake")
        set(ngraph_DIR "/opt/intel/openvino/runtime/cmake")
        find_package(InferenceEngine REQUIRED)
        find_package(ngraph REQUIRED)
        set(link ${NGRAPH_LIBRARIES} ${InferenceEngine_LIBRARIES} ${link})
    endif()
    set(include ${InferenceEngine_INCLUDE_DIRS} ${include})
    set(include ${ngraph_INCLUDE_DIRS} ${include})
endif()

#find source
file(GLOB_RECURSE src
    "src/**"
)

ament_auto_add_library(${PROJECT_NAME} SHARED ${src})
target_include_directories(${PROJECT_NAME} PUBLIC include ${include})
target_link_libraries(${PROJECT_NAME} ${link})

if(TRT AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(${PROJECT_NAME} PUBLIC "/home/cx/3rdParty/TensorRT-8.6.1.6/lib")
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_detector::DetectorNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  models
  config
  launch
)
