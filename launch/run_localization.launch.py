import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    return LaunchDescription(
        [
            params_declare,
            rviz_declare,
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments='0.0 0.0 0.0 0.0 0.0 0.0 liorf_map liorf_odom'.split(' '),
                parameters=[parameter_file],
                output='screen'
            ),
            # Node(
            #     package="liorf_localization",
            #     executable="liorf_localization_imuPreintegration",
            #     name="liorf_localization_imuPreintegration",
            #     parameters=[parameter_file],
            #     output="screen",
            # ),
            Node(
                package="liorf_localization",
                executable="liorf_localization_wheelOdomPreintegration",
                name="liorf_localization_wheelOdomPreintegration",
                parameters=[parameter_file],
                output="screen",
            ),
            Node(
                package="liorf_localization",
                executable="liorf_localization_imageProjection",
                name="liorf_localization_imageProjection",
                parameters=[parameter_file],
                output="screen",
            ),
            Node(
                package="liorf_localization",
                executable="liorf_localization_mapOptmization",
                name="liorf_localization_mapOptmization",
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
                condition=IfCondition(EqualsSubstitution(use_rviz, "true")),
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                output='screen'
            )
        ]
    )
