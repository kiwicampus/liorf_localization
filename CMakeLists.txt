cmake_minimum_required(VERSION 3.10)
project(liorf_localization)

# GTSAM on this image was built against Eigen 3.3.x (config.h:
# GTSAM_EIGEN_VERSION_WORLD=3, _MAJOR=3), but this image's system Eigen3 —
# what every other package's find_package(Eigen3) resolves to — is 3.4.0.
# Without a matching Eigen, GTSAM's own static_assert in gtsam/base/Vector.h
# ("GTSAM was built against a different version of Eigen") fails the compile.
# This must be set before ANY find_package() call below: PCL's own
# find_package(PCL REQUIRED) internally resolves and caches Eigen3 first, so
# setting this any later has no effect (CMake reuses the cached result).
# ENV{Eigen3_ROOT} (not the plain CMake variable) is used because the
# CMake-variable form of <Package>_ROOT requires policy CMP0074, which this
# file doesn't opt into (cmake_minimum_required predates it); the
# environment-variable form works regardless of that policy. This is scoped
# to this package's own CMake configure only — it has no effect on any other
# package's build even within the same colcon batch. Eigen 3.4 documents
# itself as a safe, ABI-preserving upgrade from 3.3 for standard dense
# matrix/vector usage (no exotic SIMD-width-dependent layouts are used here),
# so compiling this package against 3.3.9 instead of 3.4.0 is expected to be
# safe in practice.
set(ENV{Eigen3_ROOT} /opt/eigen-3.3.9)

# Silence upstream "CMake Warning (dev)" noise from PCL/GTSAM/OpenCV config
# files so real build errors aren't buried in push logs. This only hides the
# messages (equivalent to -Wno-dev); it does NOT change policy behavior, since
# e.g. forcing CMP0146 to NEW breaks OpenCVConfig.cmake's find_package(CUDA)
# on this CUDA 11.4 install (no CUDAConfig.cmake shipped, only the legacy
# FindCUDA.cmake module that NEW policy behavior stops using).
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -pthread")

include(cmake/rosidl_get_typesupport_target.cmake)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(initial_pose_interfaces REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)

# Kronos/GTSAM ships INTERFACE_COMPILE_OPTIONS "-march=native", which breaks
# Balena aarch64 builders (and is wrong for Jetson deploy targets anyway).
if(TARGET gtsam)
  set_property(TARGET gtsam PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()

set(ros2_lib "rclcpp;rclcpp_components;rclpy;std_msgs;sensor_msgs;geometry_msgs;cv_bridge;nav_msgs;pcl_msgs;pcl_conversions;visualization_msgs;tf2;tf2_ros;tf2_eigen;tf2_sensor_msgs;tf2_geometry_msgs;initial_pose_interfaces")

rosidl_generate_interfaces(${PROJECT_NAME} 
  "msg/CloudInfo.msg"
  "msg/OptimizationInfo.msg"
  "msg/LocalizationInfo.msg"
  "srv/SaveMap.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# include directories
include_directories(
  lib
	include
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)

# link directories
link_directories(
  lib
  include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

file(GLOB common_lib "lib/*.cpp")

###########
## Build ##
###########
# <!-- liorf_localization_yjz_lucky_boy -->

# Create a common library for shared code
add_library(liorf_common SHARED ${common_lib})
ament_target_dependencies(liorf_common ${ros2_lib})
target_link_libraries(liorf_common "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Image Projection Component Library
set(image_projection_library_name liorf_localization_imageProjection_core)
add_library(${image_projection_library_name} SHARED src/imageProjection.cpp)
ament_target_dependencies(${image_projection_library_name} ${ros2_lib})
target_link_libraries(${image_projection_library_name} liorf_common "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
rclcpp_components_register_nodes(${image_projection_library_name} "ImageProjection")

# Image Projection Standalone Executable
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
ament_target_dependencies(${PROJECT_NAME}_imageProjection ${ros2_lib})
target_link_libraries(${PROJECT_NAME}_imageProjection liorf_common "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Map Optimization Component Library
set(map_optimization_library_name liorf_localization_mapOptmization_core)
add_library(${map_optimization_library_name} SHARED src/mapOptmization.cpp)
ament_target_dependencies(${map_optimization_library_name} ${ros2_lib})
target_link_libraries(${map_optimization_library_name} liorf_common Boost::timer "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX gtsam)
rclcpp_components_register_nodes(${map_optimization_library_name} "mapOptimization")

# Map Optimization Standalone Executable
#
# Links against the component library above instead of recompiling the class
# from source: previously this executable and the library each independently
# compiled the entire (unsplit) mapOptmization.cpp, doubling the heaviest
# compile in the package for no benefit since both produced identical code.
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptimizationNode.cpp)
ament_target_dependencies(${PROJECT_NAME}_mapOptmization ${ros2_lib})
target_link_libraries(${PROJECT_NAME}_mapOptmization ${map_optimization_library_name} liorf_common Boost::timer "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX gtsam)

# IMU Preintegration - Regular Executable (not a component)
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
ament_target_dependencies(${PROJECT_NAME}_imuPreintegration ${ros2_lib})
target_link_libraries(${PROJECT_NAME}_imuPreintegration liorf_common Boost::timer "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

# WheelOdom Preintegration - Regular Executable (not a component)
add_executable(${PROJECT_NAME}_wheelOdomPreintegration src/wheelOdomPreintegration.cpp)
ament_target_dependencies(${PROJECT_NAME}_wheelOdomPreintegration ${ros2_lib})
target_link_libraries(${PROJECT_NAME}_wheelOdomPreintegration liorf_common Boost::timer "${cpp_typesupport_target}" ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)


install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(
  TARGETS
    liorf_common
    ${image_projection_library_name}
    ${map_optimization_library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    ${PROJECT_NAME}_imageProjection
    ${PROJECT_NAME}_mapOptmization
    ${PROJECT_NAME}_imuPreintegration
    ${PROJECT_NAME}_wheelOdomPreintegration
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY "include/"
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)

ament_package()