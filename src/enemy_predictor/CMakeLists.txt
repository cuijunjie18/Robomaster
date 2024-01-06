cmake_minimum_required(VERSION 3.8)
project(enemy_predictor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#平台配置
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# OpenCV
find_package(OpenCV 4 REQUIRED)
set(link ${OpenCV_LIBS} ${link})

# Eigen
find_package(Eigen3 3.3 REQUIRED)
set(link Eigen3::Eigen ${link})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
set(link ${Sophus_LIBRARIES} ${link})

# find ament_cmake
find_package(ament_cmake_auto REQUIRED)
find_package(rm_interfaces REQUIRED)
find_package(rm_utils REQUIRED)
ament_auto_find_build_dependencies()

# find src
file(GLOB_RECURSE src "src/*.cpp")
ament_auto_add_library(${PROJECT_NAME} SHARED ${src})
target_link_libraries(${PROJECT_NAME} ${link})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN enemy_predictor::EnemyPredictorNode
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
  config
  launch
)
