import os
import time
import subprocess
import threading

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.events.process.process_started import ProcessStarted
from launch.event_handlers.on_process_start import OnProcessStart
from launch.actions import RegisterEventHandler
from launch.launch_context import LaunchContext

# Try to import GcpOrLocalParamsFile, fall back to None if not available
try:
    from python_utils.launch_utils import GcpOrLocalParamsFile
    GCP_OR_LOCAL_AVAILABLE = True
except ImportError:
    GcpOrLocalParamsFile = None
    GCP_OR_LOCAL_AVAILABLE = False

def generate_launch_description():
    share_dir = get_package_share_directory("liorf_localization")
    
    # Use GcpOrLocalParamsFile if available, otherwise use LaunchConfiguration
    if GCP_OR_LOCAL_AVAILABLE:
        # Use GcpOrLocalParamsFile to handle parameter file loading
        gcp_params = GcpOrLocalParamsFile(
            env_var_name="LIORF_PARAMS_FILE",
            default_file_path=os.path.join(share_dir, "config", "localization.yaml")
        )
        parameter_file = gcp_params.local_file_path
        # Still declare the launch argument for backward compatibility
        params_declare = DeclareLaunchArgument(
            "liorf_params_file",
            default_value=parameter_file,
            description="Path to the ROS2 parameters file to use.",
        )
    else:
        # Fall back to original LaunchConfiguration approach
        parameter_file = LaunchConfiguration("liorf_params_file")
        params_declare = DeclareLaunchArgument(
            "liorf_params_file",
            default_value=os.path.join(share_dir, "config", "localization.yaml"),
            description="Path to the ROS2 parameters file to use.",
        )
    
    rviz_config_file = os.path.join(share_dir, "rviz", "localization.rviz")
    use_rviz = LaunchConfiguration("use_rviz")
    scale_livox_imu = LaunchConfiguration("scale_livox_imu")

    rviz_declare = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz"
    )

    # in case your data comes from an unmodified livox wrapper, which outputs acceleration in `g`
    # set this to true to launch a node to scale times the gravity. if you already have the latest
    # version, leave it false as the default
    livox_scale_imu_declare = DeclareLaunchArgument(
        "scale_livox_imu",
        default_value="false",
        description="Whether to scale the livox imu times the gravity"
    )

    local_launch = bool(int(os.getenv("LOCAL_LAUNCH", 0)))
    respawn_nodes = bool(int(os.getenv(key="RESPAWN_NODES", default=1)))
    respawn_delay = float(os.getenv(key="RESPAWN_DELAY", default=5))

    if local_launch:
        os.environ["LIDAR_LOCALIZATION"] = '1'

    launch_description = [
        params_declare,
        rviz_declare,
        livox_scale_imu_declare,
        Node(
            package="liorf_localization",
            executable="liorf_localization_imageProjection",
            name="liorf_localization_imageProjection",
            parameters=[parameter_file],
            output="screen",
            respawn=respawn_nodes,
            respawn_delay=respawn_delay,
            remappings=[
                ('/odometry/imu_incremental', '/wheel_odometry/global_odometry'),
            ],
        ),
        Node(
            package="liorf_localization",
            executable="liorf_localization_mapOptmization",
            name="liorf_localization_mapOptmization",
            parameters=[parameter_file],
            output="screen",
            respawn=respawn_nodes,
            respawn_delay=respawn_delay,
            remappings=[
                ('/odometry/imu_incremental', '/wheel_odometry/global_odometry'),
            ],
        ),
        # Node(
        #     package="liorf_localization",
        #     executable="liorf_localization_wheelOdomPreintegration",
        #     name="liorf_localization_wheelOdomPreintegration",
        #     parameters=[parameter_file],
        #     output="screen",
        #     respawn=respawn_nodes,
        #     respawn_delay=respawn_delay,
        # ),
    ]

    if_condition = IfCondition(scale_livox_imu)
    unless_condition = UnlessCondition(scale_livox_imu)

    livox_imu_scaler_node = Node(
        package="livox_imu_scaler",
        executable="livox_imu_scaler",
        name="livox_imu_scaler",
        parameters=[parameter_file],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        condition=if_condition,
    )

    complementary_filter_w_scaler = Node(
        package="imu_complementary_filter",
        executable="complementary_filter_node",
        name="complementary_filter_node",
        parameters=[parameter_file],
        remappings=[
            ('/imu/data', '/imu/data_livox')
        ],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        condition=if_condition,
    )

    complementary_filter_wo_scaler = Node(
        package="imu_complementary_filter",
        executable="complementary_filter_node",
        name="complementary_filter_node",
        parameters=[parameter_file],
        remappings=[
            ('/imu/data', '/imu/data_livox'),
            ('/imu/data_raw', '/livox/imu')
        ],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        condition=unless_condition,
    )

    launch_description.append(livox_imu_scaler_node)
    launch_description.append(complementary_filter_w_scaler)
    launch_description.append(complementary_filter_wo_scaler)

    if local_launch:
        launch_description.append(SetEnvironmentVariable('LIDAR_LOCALIZATION', '1'))
        launch_description.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ))
        launch_description.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0.16", "0", "0.6", "0", "0.25", "0", "base_link", "livox_link"],
            output="screen",
        ))
        launch_description.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "livox_link", "livox_frame"],
            output="screen",
        ))
        launch_description.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "gps"],
            output="screen",
        ))
        launch_description.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "inertial_link"],
            output="screen",
        ))
        # now you are meant to launch liorf through robot_localization.launch.py
        # launch_description.append(IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('location'), 'launch', 'robot_localization.launch.py')]),
        # ))

    def reniceness_execute():
        time.sleep(10)
        print(f"Renicing map optimization node in localization")
        cmd = "ps -eLf | grep 'liorf_localization_mapOptmization' | grep -v grep | awk '{print $4}' | xargs -r -n1 renice -20 -p 1> /dev/null"
        subprocess.call(cmd, shell=True)

    def reniceness_map_optimization(event: ProcessStarted, context: LaunchContext):
        # Start a new thread to run the command only if this is a restart
        if "liorf_localization_mapOptmization" in " ".join(event.action.cmd):
            threading.Thread(target=reniceness_execute).start()

    reniceness_map_optimization_event_handler = RegisterEventHandler(
        event_handler=OnProcessStart(on_start=reniceness_map_optimization)
    )

    launch_description.append(reniceness_map_optimization_event_handler)

    return LaunchDescription(launch_description)
