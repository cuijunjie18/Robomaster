cmake_minimum_required(VERSION 3.8)
project(rm_can_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Use C++17
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rm_utils REQUIRED)
find_package(rm_interfaces REQUIRED)
ament_auto_find_build_dependencies()

# find src
file(GLOB_RECURSE src "src/*.cpp")
ament_auto_add_library(${PROJECT_NAME} SHARED ${src})
ament_target_dependencies(${PROJECT_NAME} rm_interfaces)
ament_target_dependencies(${PROJECT_NAME} rm_utils)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_can_driver::RMCanDriver
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
