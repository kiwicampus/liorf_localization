import os
import time
import subprocess
import threading

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

from launch.events.process.process_started import ProcessStarted
from launch.event_handlers.on_process_start import OnProcessStart
from launch.actions import RegisterEventHandler
from launch.launch_context import LaunchContext

# Try to import GcpOrLocalParamsFile, fall back to None if not available
try:
    from python_utils.launch_utils import GcpOrLocalParamsFile, parse_bool2string

    GCP_OR_LOCAL_AVAILABLE = True
except ImportError:
    GcpOrLocalParamsFile = None
    parse_bool2string = lambda x: str(bool(x)).lower()
    GCP_OR_LOCAL_AVAILABLE = False


def generate_launch_description():
    share_dir = get_package_share_directory("liorf_localization")

    # Launch arguments
    use_composition = LaunchConfiguration("use_composition")
    container_name = LaunchConfiguration("container_name")
    use_respawn = LaunchConfiguration("use_respawn")

    # Use GcpOrLocalParamsFile if available, otherwise use LaunchConfiguration
    if GCP_OR_LOCAL_AVAILABLE:
        # Use GcpOrLocalParamsFile to handle parameter file loading
        gcp_params = GcpOrLocalParamsFile(
            env_var_name="LIORF_PARAMS_FILE",
            default_file_path=os.path.join(share_dir, "config", "localization.yaml"),
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

    rviz_declare = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Whether to launch RViz"
    )

    is_relaunch = LaunchConfiguration("is_relaunch")
    is_relaunch_declare = DeclareLaunchArgument(
        "is_relaunch",
        default_value="false",
        description="Whether this is a relaunch (used to avoid re-launching non-composable nodes)",
    )

    local_launch = bool(int(os.getenv("LOCAL_LAUNCH", 0)))
    respawn_nodes = bool(int(os.getenv(key="RESPAWN_NODES", default=1)))
    respawn_delay = float(os.getenv(key="RESPAWN_DELAY", default=5))

    if local_launch:
        os.environ["LIDAR_LOCALIZATION"] = "1"

    # Composable nodes for ImageProjection and mapOptimization
    composable_liorf_nodes = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=[
            ComposableNode(
                package="liorf_localization",
                plugin="ImageProjection",
                name="liorf_localization_imageProjection",
                parameters=[parameter_file],
                remappings=[
                    ("/odometry/imu_incremental", "/wheel_odometry/global_odometry"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="liorf_localization",
                plugin="mapOptimization",
                name="liorf_localization_mapOptmization",
                parameters=[parameter_file],
                remappings=[
                    ("/odometry/imu_incremental", "/wheel_odometry/global_odometry"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        condition=IfCondition(use_composition),
    )

    # Standalone nodes (when composition is disabled)
    standalone_image_projection = Node(
        package="liorf_localization",
        executable="liorf_localization_imageProjection",
        name="liorf_localization_imageProjection",
        parameters=[parameter_file],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        remappings=[
            ("/odometry/imu_incremental", "/wheel_odometry/global_odometry"),
        ],
        condition=UnlessCondition(use_composition),
    )

    standalone_map_optimization = Node(
        package="liorf_localization",
        executable="liorf_localization_mapOptmization",
        name="liorf_localization_mapOptmization",
        # prefix="valgrind --tool=massif",
        parameters=[parameter_file],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        remappings=[
            ("/odometry/imu_incremental", "/wheel_odometry/global_odometry"),
        ],
        condition=UnlessCondition(use_composition),
    )

    launch_description = [
        # Launch arguments
        DeclareLaunchArgument(
            "use_composition",
            default_value="false",
            description="Whether to use node composition for liorf nodes",
        ),
        DeclareLaunchArgument(
            "container_name",
            default_value="localization_kronos",
            description="Name of the container to load composable nodes into",
        ),
        params_declare,
        rviz_declare,
        is_relaunch_declare,
        # Composable nodes
        composable_liorf_nodes,
        # Standalone nodes
        standalone_image_projection,
        standalone_map_optimization,
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

    imu_complementary_filter = Node(
        package="imu_complementary_filter",
        executable="complementary_filter_node",
        name="complementary_filter_node",
        parameters=[parameter_file],
        remappings=[("/imu/data", "/imu/data_livox"), ("/imu/data_raw", "/livox/imu")],
        output="screen",
        respawn=respawn_nodes,
        respawn_delay=respawn_delay,
        condition=UnlessCondition(is_relaunch),
    )

    launch_description.append(imu_complementary_filter)

    if local_launch:
        launch_description.append(SetEnvironmentVariable("LIDAR_LOCALIZATION", "1"))
        launch_description.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config_file],
                output="screen",
            )
        )
        launch_description.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                arguments=[
                    "0.16",
                    "0",
                    "0.6",
                    "0",
                    "0.25",
                    "0",
                    "base_link",
                    "livox_link",
                ],
                output="screen",
            )
        )
        launch_description.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "livox_link", "livox_frame"],
                output="screen",
            )
        )
        launch_description.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "gps"],
                output="screen",
            )
        )
        launch_description.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "inertial_link"],
                output="screen",
            )
        )
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
