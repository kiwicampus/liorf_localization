import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EqualsSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    share_dir = get_package_share_directory("liorf_localization")
    parameter_file = LaunchConfiguration("params_file")
    rviz_config_file = os.path.join(share_dir, "rviz", "localization.rviz")
    use_rviz = LaunchConfiguration("use_rviz")

    params_declare = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(share_dir, "config", "localization.yaml"),
        description="Path to the ROS2 parameters file to use.",
    )

    rviz_declare = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz"
    )

    local_launch = bool(int(os.getenv("LOCAL_LAUNCH", 0)))

    if local_launch:
        os.environ["LIDAR_LOCALIZATION"] = '1'

    launch_description = [
        params_declare,
        rviz_declare,
        Node(
            package="liorf_localization",
            executable="liorf_localization_imageProjection",
            name="liorf_localization_imageProjection",
            parameters=[parameter_file],
            output="screen",
        ),
        Node(
            package="liorf_localization",
            executable="liorf_localization_mapOptimization",
            name="liorf_localization_mapOptimization",
            parameters=[parameter_file],
            # prefix=["valgrind --tool=callgrind --instr-atstart=no"],
            output="screen",
        ),
        Node(
            package="livox_imu_scaler",
            executable="livox_imu_scaler",
            name="livox_imu_scaler",
            parameters=[parameter_file],
            output="screen",
        ),
        Node(
            package="imu_complementary_filter",
            executable="complementary_filter_node",
            name="complementary_filter_node",
            parameters=[parameter_file],
            remappings=[
                ('/imu/data', '/imu/data_livox')
            ],
            output="screen",
        ),
        Node(
                package="liorf_localization",
                executable="liorf_localization_wheelOdomPreintegration",
                name="liorf_localization_wheelOdomPreintegration",
                parameters=[parameter_file],
                output="screen",
            ),
    ]

    if local_launch:
        launch_description.append(SetEnvironmentVariable('LIDAR_LOCALIZATION', '1'))
        launch_description.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ))
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0.16", "0", "0.6", "0", "0.25", "0", "base_link", "livox_link"],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0.16", "0", "0.6", "0", "0.25", "0", "base_link", "laser_link"],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "gps"],
            output="screen",
        ),
        launch_description.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('location'), 'launch', 'robot_localization.launch.py')]),
        ))

    return LaunchDescription(launch_description)

