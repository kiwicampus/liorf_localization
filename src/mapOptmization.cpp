#include "utility.h"
#include "nanoflann_pcl.h"
#include "liorf_localization/msg/cloud_info.hpp"
#include "liorf_localization/msg/localization_info.hpp"
#include "liorf_localization/srv/save_map.hpp"
#include "std_msgs/msg/empty.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>


#include "initial_pose_interfaces/srv/set_initial_pose.hpp"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    rclcpp::Subscription<liorf_localization::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initial_pose;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubGlobalMap;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;
    rclcpp::Publisher<liorf_localization::msg::CloudInfo>::SharedPtr pubSLAMInfo;

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pubMapPose;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pubGpsPose;
    rclcpp::Publisher<liorf_localization::msg::LocalizationInfo>::SharedPtr pubLocalizationInfo;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pubLocalizationRestored;
    liorf_localization::msg::LocalizationInfo msgLocalizationInfo;

    rclcpp::Service<liorf_localization::srv::SaveMap>::SharedPtr srvSaveMap;

    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    liorf_localization::msg::CloudInfo cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    nanoflann::KdTreeFLANN<PointType> kdtreeSurfFromMap;

    nanoflann::KdTreeFLANN<PointType> kdtreeSurroundingKeyPoses;
    nanoflann::KdTreeFLANN<PointType> kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterLocalMapSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::MatrixXf matP;

    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;


    // add by yjz_lucky_boy
    // localization
    bool has_global_map = false;
    bool has_initialize_pose = false;
    bool system_initialized = false;
    float initialize_pose[6];
    
    // Map deduplication
    size_t last_map_size_ = 0;
    size_t last_map_hash_ = 0;

    std::unique_ptr<tf2_ros::TransformBroadcaster> br;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    Eigen::Affine3f base_link_to_livox_, livox_to_base_link_;

    int initial_guess_max_iters_ = 0;
    int initial_guess_seconds_between_attempts_ = 10;
    int initial_guess_iters_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    bool use_map_server_ = true;
    // dynamic parameters
    // Parameters callback
    OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;

    float orientation_change_threshold_ = 0.02;
    float high_orientation_covariance_ = 9999.0;
    float prev_yaw_ = 0.0;
    float icp_fitness_score_threshold_ = 0.9;

    rclcpp::Service<initial_pose_interfaces::srv::SetInitialPose>::SharedPtr srv_set_initial_pose_;

    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("liorf_localization_mapOptimization", options)
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        subCloud = create_subscription<liorf_localization::msg::CloudInfo>("liorf_localization/deskew/cloud_info", QosPolicy(history_policy, reliability_policy),
                    std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(gpsTopic, QosPolicy(history_policy, reliability_policy),
                    std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        sub_initial_pose = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", QosPolicy(history_policy, reliability_policy),
                    std::bind(&mapOptimization::initialposeHandler, this, std::placeholders::_1));

        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/trajectory", QosPolicy(history_policy, reliability_policy));
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/map_global", QosPolicy(history_policy, reliability_policy));
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("liorf_localization/mapping/odometry", QosPolicy(history_policy, reliability_policy));
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>("liorf_localization/mapping/odometry_incremental", QosPolicy(history_policy, reliability_policy));
        pubPath = create_publisher<nav_msgs::msg::Path>("liorf_localization/mapping/path", QosPolicy(history_policy, reliability_policy));
        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/icp_loop_closure_history_cloud", QosPolicy(history_policy, reliability_policy));
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/icp_loop_closure_corrected_cloud", QosPolicy(history_policy, reliability_policy));
        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/map_local", QosPolicy(history_policy, reliability_policy));
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/cloud_registered", QosPolicy(history_policy, reliability_policy));
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/mapping/cloud_registered_raw", QosPolicy(history_policy, reliability_policy));
        pubSLAMInfo = create_publisher<liorf_localization::msg::CloudInfo>("liorf_localization/mapping/slam_info", QosPolicy(history_policy, reliability_policy));
        
        pubMapPose = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("liorf_localization/mapping/map_pose", QosPolicy(history_policy, reliability_policy));
        pubGpsPose = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("liorf_localization/mapping/gps_pose", QosPolicy(history_policy, reliability_policy));
        pubLocalizationInfo = create_publisher<liorf_localization::msg::LocalizationInfo>("liorf_localization/mapping/localization_info", QosPolicy(history_policy, reliability_policy));
        pubLocalizationRestored = create_publisher<std_msgs::msg::Empty>("liorf_localization/mapping/localization_restored", QosPolicy(history_policy, reliability_policy));

        rclcpp::PublisherOptionsWithAllocator<std::allocator<void>> pub_options;
        pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
        // Using volatile durability (default) to enable intra-process communication
        pubGlobalMap = create_publisher<sensor_msgs::msg::PointCloud2>("liorf_localization/localization/global_map", rclcpp::QoS(10).reliable(), pub_options);


        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        // Initialize the TF buffer and listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock(), tf2::Duration(tf2::BUFFER_CORE_DEFAULT_CACHE_TIME), std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*) {}));
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*) {}), false); // Pass false to not use dynamic transform topic

        // srvSaveMap = create_service<liorf_localization::srv::save_map>("liorf_localization/save_map", 
        //                 std::bind(&mapOptimization::saveMapService, this, std::placeholders::_1, std::placeholders::_2 ));

        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
        loadGlobalMap();
    
        initial_guess_max_iters_ = std::getenv("INITIAL_GUESS_MAX_ITERS")
                                    ? std::stoi(std::getenv("INITIAL_GUESS_MAX_ITERS"))
                                    : 0;
        initial_guess_seconds_between_attempts_ = std::getenv("INITIAL_GUESS_SECONDS_BETWEEN_ATTEMPTS")
                                    ? std::stoi(std::getenv("INITIAL_GUESS_SECONDS_BETWEEN_ATTEMPTS"))
                                    : 10;

        // Initial guess action client
        this->declare_parameter<bool>("use_map_server", true);
        this->get_parameter("use_map_server", use_map_server_);
        // dynamic parameters
        this->declare_parameter<float>("orientation_change_threshold", orientation_change_threshold_);
        this->declare_parameter<float>("high_orientation_covariance", high_orientation_covariance_);
        this->declare_parameter<float>("icp_fitness_score_threshold", icp_fitness_score_threshold_);

        this->get_parameter("orientation_change_threshold", orientation_change_threshold_);
        this->get_parameter("high_orientation_covariance", high_orientation_covariance_);
        this->get_parameter("icp_fitness_score_threshold", icp_fitness_score_threshold_);
        params_callback_handle_ = this->add_on_set_parameters_callback(std::bind(&mapOptimization::on_parameters_set_callback, this, std::placeholders::_1));

        srv_set_initial_pose_ = create_service<initial_pose_interfaces::srv::SetInitialPose>(
            "set_initial_pose",
            std::bind(&mapOptimization::setInitialPoseCallback, this, std::placeholders::_1, std::placeholders::_2));
        
        livox_to_base_link_ = Eigen::Matrix4f::Identity();
        base_link_to_livox_ = Eigen::Matrix4f::Identity();
    }


    rcl_interfaces::msg::SetParametersResult on_parameters_set_callback(
        const std::vector<rclcpp::Parameter>& parameters){
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        for (const auto& param : parameters) {
            if (param.get_name() == "orientation_change_threshold") {
                orientation_change_threshold_ = param.as_double();
                RCLCPP_INFO(this->get_logger(), "orientation_change_threshold updated to %f", orientation_change_threshold_);
            }
            else if (param.get_name() == "high_orientation_covariance") {
                high_orientation_covariance_ = param.as_double();
                RCLCPP_INFO(this->get_logger(), "high_orientation_covariance updated to %f", high_orientation_covariance_);
            }
            else if (param.get_name() == "icp_fitness_score_threshold") {
                icp_fitness_score_threshold_ = param.as_double();
                RCLCPP_INFO(this->get_logger(), "icp_fitness_score_threshold updated to %f", icp_fitness_score_threshold_);
            }
        }
        return result;
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        // kdtreeSurroundingKeyPoses.reset(new nanoflann::KdTreeFLANN<PointType>());
        // kdtreeHistoryKeyPoses.reset(new nanoflann::KdTreeFLANN<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        // kdtreeSurfFromMap.reset(new nanoflann::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = Eigen::MatrixXf::Zero(6, 6);
    }

    // add by yjz_lucky_boy
    void loadGlobalMap()
    {
        RCLCPP_INFO(get_logger(), "using map server: %d", use_map_server_);
        if (use_map_server_) {
            // Set up subscription with volatile durability (default) to enable intra-process communication
            // The map server will auto-republish when this subscriber joins
            rclcpp::QoS map_qos(1);
            map_qos.reliable();
            rclcpp::SubscriptionOptions sub_options;
            sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

            map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                "/pointcloud_map", map_qos,
                [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                    RCLCPP_INFO(get_logger(), "Received map from pointcloud map server");
                    processReceivedMap(msg);
                },
                sub_options);
            RCLCPP_INFO(get_logger(), "Waiting for map from pointcloud map server...");
        } else {
            // Load from file
            std::string global_map = savePCDDirectory + "GlobalMap.pcd";
            RCLCPP_INFO(get_logger(), "Loading map from file: %s", global_map.c_str());
            
            pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
            if (pcl::io::loadPCDFile<PointType>(global_map, *laserCloudSurfFromMap) == -1) {
                RCLCPP_ERROR(get_logger(), "Failed to load PCD file: %s", global_map.c_str());
                has_global_map = false;
                return;
            }
            processLoadedMap(laserCloudSurfFromMap);
            publishCloud(pubGlobalMap, laserCloudSurfFromMapDS, this->now(), mapFrame, false); 
        }
    }

    void processReceivedMap(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Check if this is the same map we already have (deduplication)
        size_t current_size = msg->width * msg->height;
        size_t current_hash = std::hash<std::string>{}(
            std::string(reinterpret_cast<const char*>(msg->data.data()), 
                       std::min(msg->data.size(), size_t(1024)))  // Hash first 1KB for speed
        );
        
        if (has_global_map && current_size == last_map_size_ && current_hash == last_map_hash_) {
            RCLCPP_DEBUG(get_logger(), "Received identical map, skipping reprocessing");
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Processing new map (size: %zu points)", current_size);
        last_map_size_ = current_size;
        last_map_hash_ = current_hash;
        
        pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*msg, *laserCloudSurfFromMap);
        processLoadedMap(laserCloudSurfFromMap);
    }

    void processLoadedMap(pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap)
    {
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        std::cout << "global map size: " << laserCloudSurfFromMapDSNum << std::endl;

        if (laserCloudSurfFromMapDSNum < 1000) {
            RCLCPP_WARN(get_logger(), "Map has too few points (%d < 1000)", laserCloudSurfFromMapDSNum);
            has_global_map = false;
            return;
        }
        
        has_global_map = true;

        kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapDS);

        sleep(3);
    }

    // add by yjz_lucky_boy
    void initialposeHandler(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msgIn) 
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            // Lookup transform from base_link to livox_link
            transformStamped = tf_buffer_->lookupTransform("livox_link", "base_link", tf2::TimePointZero);

            // Convert TransformStamped to Affine3f
            tf2::Quaternion quat;
            tf2::fromMsg(transformStamped.transform.rotation, quat);
            double roll, pitch, yaw;
            tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

            // Construct Affine3f transformation
            base_link_to_livox_ = Eigen::Affine3f::Identity();
            base_link_to_livox_.translate(Eigen::Vector3f(transformStamped.transform.translation.x, transformStamped.transform.translation.y, transformStamped.transform.translation.z));
            base_link_to_livox_.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
            base_link_to_livox_.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
            base_link_to_livox_.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));

            livox_to_base_link_ = base_link_to_livox_.inverse();

            std::cout << "livox to base_link is\n" << livox_to_base_link_.matrix() << std::endl;

        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform base_link to livox_link: %s", ex.what());
            return;
        }
        // in case you want to republish the map if it was not loaded on rviz
        // publishCloud(pubGlobalMap, laserCloudSurfFromMapDS, rclcpp::Time(), mapFrame);
        tf2::Quaternion q(msgIn->pose.pose.orientation.x, msgIn->pose.pose.orientation.y, 
                            msgIn->pose.pose.orientation.z, msgIn->pose.pose.orientation.w);
        tf2::Matrix3x3 qm(q);

        double roll, pitch, yaw;
        qm.getRPY(roll, pitch, yaw);

        initialize_pose[0] = roll;
        initialize_pose[1] = pitch;
        initialize_pose[2] = yaw;

        initialize_pose[3] = msgIn->pose.pose.position.x;
        initialize_pose[4] = msgIn->pose.pose.position.y;
        initialize_pose[5] = msgIn->pose.pose.position.z;

        std::cout << "manual initialize pose: \n" << initialize_pose[3] << "\n" << initialize_pose[4] << "\n" << initialize_pose[5] << "\n" 
                  << initialize_pose[0] << "\n" << initialize_pose[1] << "\n" << initialize_pose[2] << std::endl;

        has_initialize_pose = true;
        system_initialized = false;
    }

    void laserCloudInfoHandler(const liorf_localization::msg::CloudInfo::SharedPtr msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = ROS_TIME(msgIn->header.stamp);

        // extract info and feature cloud
        cloudInfo = *msgIn;

        std::lock_guard<std::mutex> lock(mtx);  // Lock for the entire handler

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudSurfLast);

            adjustForRotation();

            if (!system_initialized)
              if(!systemInitialize())
                return;

            updateInitialGuess();

            downsampleCurrentScan();

            auto start = std::chrono::high_resolution_clock::now();
            bool optimization_success = scan2MapOptimization();
            if(optimization_success)
            {
                saveKeyFramesAndFactor();

                correctPoses();

                publishOdometry();

                publishFrames();
                msgLocalizationInfo.optimization_info.optimization_timed_out = false;
            }
            else
            {
                msgLocalizationInfo.optimization_info.optimization_timed_out = true;
                auto end = std::chrono::high_resolution_clock::now();
                RCLCPP_WARN(get_logger(), "Cloud handling timed out after %f seconds. dropping this scan", (end - start).count());
            }


            // End time measurement
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            // Log the elapsed time
            msgLocalizationInfo.optimization_info.optimization_time = static_cast<float>(elapsed.count());
            pubLocalizationInfo->publish(msgLocalizationInfo);
            // RCLCPP_INFO(rclcpp::get_logger("laserCloudInfoHandler"), "Execution time: %f seconds", elapsed.count());
        }

    }

    bool systemInitialize()
    {
        if (!has_global_map)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Liorf has not received global map yet.");
            return false;
        }

        if(!has_initialize_pose)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "need initilize pose from rviz.");
            msgLocalizationInfo.initial_pose_accepted = false;
            publishThrottle<liorf_localization::msg::LocalizationInfo>(pubLocalizationInfo, msgLocalizationInfo, 10.0);
            return false;
        }

        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(20);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        Eigen::Affine3f initialize_affine = trans2Affine3f(initialize_pose);

        std::cout << "initialize_affine: \n" << initialize_affine.matrix() << std::endl;

        pcl::PointCloud<PointType>::Ptr out_cloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laserCloudSurfLast, *out_cloud, initialize_affine);

        std::cout << "average x in transformed point cloud: " << out_cloud->points[out_cloud->size() / 2].x << " and average x in original point cloud: " << laserCloudSurfLast->points[laserCloudSurfLast->size() / 2].x << std::endl;
        
        // Crop the reference map around the target pose for faster ICP alignment
        pcl::PointCloud<PointType>::Ptr cropped_map(new pcl::PointCloud<PointType>());
        pcl::CropBox<PointType> crop_box;
        
        // Set crop box around the target pose (initialize_pose[3], initialize_pose[4], initialize_pose[5])
        // Use surroundingKeyframeSearchRadius as the crop radius
        float crop_radius = surroundingKeyframeSearchRadius;
        float min_x = initialize_pose[3] - crop_radius;
        float max_x = initialize_pose[3] + crop_radius;
        float min_y = initialize_pose[4] - crop_radius;
        float max_y = initialize_pose[4] + crop_radius;
        float min_z = initialize_pose[5] - crop_radius;
        float max_z = initialize_pose[5] + crop_radius;
        
        crop_box.setMin(Eigen::Vector4f(min_x, min_y, min_z, 1.0));
        crop_box.setMax(Eigen::Vector4f(max_x, max_y, max_z, 1.0));
        crop_box.setInputCloud(laserCloudSurfFromMapDS);
        crop_box.filter(*cropped_map);
        
        RCLCPP_INFO(get_logger(), "Cropped map from %d to %d points (radius: %.1f m)", 
                   laserCloudSurfFromMapDS->size(), cropped_map->size(), crop_radius);
        
        // Check if we have enough points in the cropped map
        if (cropped_map->size() < 1000) {
            RCLCPP_WARN(get_logger(), "Cropped map has too few points (%d < 1000), using full map", cropped_map->size());
            cropped_map = laserCloudSurfFromMapDS; // Fallback to full map
        }
        
        // Align clouds using the cropped map
        icp.setInputSource(out_cloud);
        icp.setInputTarget(cropped_map);
        icp.align(*result);

        Eigen::Affine3f correctionLidarFrame;
        float x, y, z, roll, pitch, yaw;
        correctionLidarFrame = icp.getFinalTransformation();
        Eigen::Affine3f tCorrect = correctionLidarFrame * initialize_affine;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

        transformTobeMapped[0] = roll;
        transformTobeMapped[1] = pitch;
        transformTobeMapped[2] = yaw;
        transformTobeMapped[3] = x;
        transformTobeMapped[4] = y;
        transformTobeMapped[5] = z;

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        *cloudOut += *transformPointCloud(laserCloudSurfLast, &thisPose6D);
        publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, mapFrame);

        RCLCPP_INFO(get_logger(), "icp.hasConverged(): %d", icp.hasConverged());
        RCLCPP_INFO(get_logger(), "icp.getFitnessScore(): %f", icp.getFitnessScore());

        if (icp.hasConverged() && icp.getFitnessScore() < icp_fitness_score_threshold_)
        {
            RCLCPP_INFO(get_logger(), "initialize pose sucessful");
            system_initialized = true;
            msgLocalizationInfo.initial_pose_accepted = true;
            RCLCPP_INFO(get_logger(), "initializing pose accepted, original pose: %f, %f, yaw: %f, refined pose: %f, %f, yaw: %f. used pointcloud with %d points", 
                        initialize_pose[3], initialize_pose[4], initialize_pose[2], x, y, yaw, laserCloudSurfLast->size());
            pubLocalizationRestored->publish(std_msgs::msg::Empty());
            return true;
        } 
        else
        {
            RCLCPP_ERROR(get_logger(), "initialize pose failed");
            has_initialize_pose = false;
            system_initialized = false;
            return false;
        }
    }

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        // Odometry message is assumed to be already converted to the map frame
        // Simply publish it and add to queue
        if (!useGpsElevation)
        {
            odomMsg->pose.pose.position.z = transformTobeMapped[5];
            odomMsg->pose.covariance[14] = 0.01;
        }
        publishPoseWithCovariance(pubGpsPose, odomMsg->pose, odomMsg->header.stamp, mapFrame);
        gpsQueue.push_back(*odomMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(const std::shared_ptr<liorf_localization::srv::SaveMap::Request> req,
                                std::shared_ptr<liorf_localization::srv::SaveMap::Response> res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req->destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req->destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map

      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req->resolution != 0)
      {
        cout << "\n\nSave resolution: " << req->resolution << endl;
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {

        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res->success = ret == 0;

      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        rclcpp::Rate rate(0.2);
        while (rclcpp::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        std::shared_ptr<liorf_localization::srv::SaveMap::Request> req = std::make_unique<liorf_localization::srv::SaveMap::Request>();
        std::shared_ptr<liorf_localization::srv::SaveMap::Response> res = std::make_unique<liorf_localization::srv::SaveMap::Response>();

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround->get_subscription_count() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        nanoflann::KdTreeFLANN<PointType> kdtreeGlobalMap;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap.setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap.radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap.nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (common_lib_->pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, mapFrame);
    }

    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        if (cloudKeyPoses3D->points.empty())
        {
            // transformTobeMapped[0] = cloudInfo.imurollinit;
            // transformTobeMapped[1] = cloudInfo.imupitchinit;
            // transformTobeMapped[2] = cloudInfo.imuyawinit;

            // if (!useImuHeadingInitialization)
            //     transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imurollinit, cloudInfo.imupitchinit, cloudInfo.imuyawinit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomavailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialguessx,    cloudInfo.initialguessy,     cloudInfo.initialguessz, 
                                                               cloudInfo.initialguessroll, cloudInfo.initialguesspitch, cloudInfo.initialguessyaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imurollinit, cloudInfo.imupitchinit, cloudInfo.imuyawinit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuavailable == true && imuType)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imurollinit, cloudInfo.imupitchinit, cloudInfo.imuyawinit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imurollinit, cloudInfo.imupitchinit, cloudInfo.imuyawinit); // save imu before return;
            return;
        }
    }


    void downsampleCurrentScan()
    {
        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void adjustForRotation()
    {
        // Use the cached transform to adjust the cloud from livox_link to base_link
        pcl::transformPointCloud(*laserCloudSurfLast, *laserCloudSurfLast, livox_to_base_link_);
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap.nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[2]);
        float crx = cos(transformTobeMapped[2]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        Eigen::MatrixXf matA = Eigen::MatrixXf::Zero(laserCloudSelNum, 6);
        Eigen::MatrixXf matAt = Eigen::MatrixXf::Zero(6, laserCloudSelNum);
        Eigen::MatrixXf matAtA = Eigen::MatrixXf::Zero(6, 6);
        Eigen::VectorXf matB = Eigen::VectorXf::Zero(laserCloudSelNum);
        Eigen::VectorXf matAtB = Eigen::VectorXf::Zero(6);
        Eigen::VectorXf matX = Eigen::VectorXf::Zero(6);

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].x;
            pointOri.y = laserCloudOri->points[i].y;
            pointOri.z = laserCloudOri->points[i].z;
            // lidar -> camera
            coeff.x = coeffSel->points[i].x;
            coeff.y = coeffSel->points[i].y;
            coeff.z = coeffSel->points[i].z;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
/*             float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
             */

            float arx = (-srx * cry * pointOri.x - (srx * sry * srz + crx * crz) * pointOri.y + (crx * srz - srx * sry * crz) * pointOri.z) * coeff.x
                      + (crx * cry * pointOri.x - (srx * crz - crx * sry * srz) * pointOri.y + (crx * sry * crz + srx * srz) * pointOri.z) * coeff.y;

            float ary = (-crx * sry * pointOri.x + crx * cry * srz * pointOri.y + crx * cry * crz * pointOri.z) * coeff.x
                      + (-srx * sry * pointOri.x + srx * sry * srz * pointOri.y + srx * cry * crz * pointOri.z) * coeff.y
                      + (-cry * pointOri.x - sry * srz * pointOri.y - sry * crz * pointOri.z) * coeff.z;

            float arz = ((crx * sry * crz + srx * srz) * pointOri.y + (srx * crz - crx * sry * srz) * pointOri.z) * coeff.x
                      + ((-crx * srz + srx * sry * crz) * pointOri.y + (-srx * sry * srz - crx * crz) * pointOri.z) * coeff.y
                      + (cry * crz * pointOri.y - cry * srz * pointOri.z) * coeff.z;

            // camera -> lidar
            matA(i, 0) = arz;
            matA(i, 1) = ary;
            matA(i, 2) = arx;
            matA(i, 3) = coeff.x;
            matA(i, 4) = coeff.y;
            matA(i, 5) = coeff.z;
            matB(i) = -coeff.intensity;
        }

        matAt = matA.transpose();
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        matX = matAtA.colPivHouseholderQr().solve(matAtB);

        if (iterCount == 0) {

            Eigen::VectorXf matE = Eigen::VectorXf::Zero(6);
            Eigen::MatrixXf matV = Eigen::MatrixXf::Zero(6, 6);
            Eigen::MatrixXf matV2 = Eigen::MatrixXf::Zero(6, 6);

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(matAtA);
            matE = eig.eigenvalues().real();
            matV = eig.eigenvectors().real();
            matV2 = matV;

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE(i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inverse() * matV2;
        }

        if (isDegenerate)
        {
            Eigen::VectorXf matX2 = Eigen::VectorXf::Zero(6);
            matX2 = matP * matX;
            matX = matX2;
        }

        transformTobeMapped[0] += matX(0);
        transformTobeMapped[1] += matX(1);
        transformTobeMapped[2] += matX(2);
        transformTobeMapped[3] += matX(3);
        transformTobeMapped[4] += matX(4);
        transformTobeMapped[5] += matX(5);

        // Compute the residual vector
        Eigen::VectorXf matR = matA * matX + matB;

        // Calculate the RMSE
        double rmse = sqrt(matR.squaredNorm() / matR.size());

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX(0)), 2) +
                            pow(pcl::rad2deg(matX(1)), 2) +
                            pow(pcl::rad2deg(matX(2)), 2));
        float deltaT = sqrt(
                            pow(matX(3) * 100, 2) +
                            pow(matX(4) * 100, 2) +
                            pow(matX(5) * 100, 2));

        msgLocalizationInfo.optimization_info.optimization_residuals_rmse = rmse;
        msgLocalizationInfo.optimization_info.optimization_delta_r = deltaR;
        msgLocalizationInfo.optimization_info.optimization_delta_t = deltaT;

        if (deltaR < static_cast<double>(mappingLmConvergenceRot) && deltaT < static_cast<double>(mappingLmConvergenceTrans)) {
            return true; // converged
        }
        return false; // keep optimizing
    }
    // <!-- liorf_localization_yjz_lucky_boy -->
    bool scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        auto startTime = std::chrono::high_resolution_clock::now();
        auto timeout = std::chrono::milliseconds(mappingProcessingTimeoutMs);
        int iterCount = 0;
        msgLocalizationInfo.tracked_features = laserCloudSurfLastDSNum;
        if (laserCloudSurfLastDSNum > 30)
        {
            
            // kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            for (iterCount = 0; iterCount < static_cast<int>(maxNumOptimizationIterations); iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
                if (elapsedTime > timeout)
                {
                    RCLCPP_WARN(get_logger(), "scan2MapOptimization timed out on iteration %i. avg iteration time was %d ms", iterCount, static_cast<float>(elapsedTime.count())/iterCount);
                    msgLocalizationInfo.optimization_info.optimization_iterations = iterCount;
                    return false;
                }
            }

            transformUpdate();
        } else {
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d planar features available.", laserCloudSurfLastDSNum);
        }
        msgLocalizationInfo.optimization_info.optimization_iterations = iterCount;
        return true;
    }

    void transformUpdate()
    {
        if (cloudInfo.imuavailable == true && imuType)
        {
            if (std::abs(cloudInfo.imupitchinit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imurollinit, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imupitchinit, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (common_lib_->pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (ROS_TIME(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (ROS_TIME(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Check if GPS measurement is too far from current pose estimate (2D distance only, ignore altitude)
                float dx = gps_x - transformTobeMapped[3];
                float dy = gps_y - transformTobeMapped[4];
                float distance = sqrt(dx * dx + dy * dy);
                if (distance > mappingGpsDistanceThreshold)
                {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, 
                        "GPS measurement rejected: too far from current estimate (%.2f m > %.2f m threshold)", 
                        distance, mappingGpsDistanceThreshold);
                    continue;
                }

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (common_lib_->pointDistance(curGPSPoint, lastGPSPoint) < mappingGpsAddingDist)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, mappingGpsCovariance), max(noise_y, mappingGpsCovariance), max(noise_z, mappingGpsCovariance);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                RCLCPP_INFO(get_logger(), "Added GPS factor");
                aLoopIsClosed = true;
                break;
            }
        }
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << poseCovariance << endl << endl;

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        rclcpp::Time t(static_cast<uint32_t>(pose_in.time * 1e9));
        pose_stamped.header.stamp = t;
        pose_stamped.header.frame_id = mapFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::msg::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = mapFrame;
        laserOdometryROS.child_frame_id = lidarFrame;
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.covariance = matrixToArray(poseCovariance, 5.0, 0.5);

        // Ref: http://wiki.ros.org/tf2/Tutorials/Migration/DataConversions
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometryROS.pose.pose.orientation = quat_msg;

        //check change in rpy
        if (std::abs(transformTobeMapped[2] - prev_yaw_) > orientation_change_threshold_) {
            // high orientation covariance
            laserOdometryROS.pose.covariance[35] = high_orientation_covariance_;
            laserOdometryROS.pose.covariance[29] = high_orientation_covariance_;
            laserOdometryROS.pose.covariance[23] = high_orientation_covariance_;
        }
        prev_yaw_ = transformTobeMapped[2];

        pubLaserOdometryGlobal->publish(laserOdometryROS);
        publishPoseWithCovariance(pubMapPose, laserOdometryROS.pose, timeLaserInfoStamp, mapFrame);

        // now, set the real covariance to the localization info message
        msgLocalizationInfo.map_pose.pose = laserOdometryROS.pose.pose;
        msgLocalizationInfo.map_pose.covariance = matrixToArray(poseCovariance, std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

        // we will let the rest of the stack handle the tranforms        
        // Publish TF
        // quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        // tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
        // tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, mapFrame);
        // geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
        // tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
        // trans_odom_to_lidar.child_frame_id = lidarFrame;
        // br->sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuavailable == true && imuType)
            {
                if (std::abs(cloudInfo.imupitchinit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imurollinit, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imupitchinit, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = mapFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            geometry_msgs::msg::Quaternion quat_msg;
            tf2::convert(quat_tf, quat_msg);
            laserOdomIncremental.pose.pose.orientation = quat_msg;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental->publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        // publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, mapFrame);
        // Publish surrounding key frames
        // publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, mapFrame);
        // publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, mapFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, mapFrame);
        }
        // publish path
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = mapFrame;
            pubPath->publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo->get_subscription_count() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                // liorf_localization::msg::CloudInfo slamInfo;
                // slamInfo.header.stamp = timeLaserInfoStamp;
                // pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                // *cloudOut += *laserCloudSurfLastDS;
                // slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                // slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, mapFrame);
                // pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                // *localMapOut += *laserCloudSurfFromMapDS;
                // slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, mapFrame);
                // pubSLAMInfo.publish(slamInfo);
                // lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }

    template<typename T>
    bool publishThrottle(const typename rclcpp::Publisher<T>::SharedPtr& pub, const T& msg, double min_interval)
    {
        static std::unordered_map<std::string, rclcpp::Time> last_pub_time;
        auto now = this->now();
        auto topic = pub->get_topic_name();
        if (last_pub_time.find(topic) == last_pub_time.end()) {
            last_pub_time[topic] = now - rclcpp::Duration::from_seconds(min_interval);
        }
        if ((now - last_pub_time[topic]).seconds() >= min_interval) {
            pub->publish(msg);
            last_pub_time[topic] = now;
            return true;
        }
        return false;
    }

    void setInitialPoseCallback(
        const std::shared_ptr<initial_pose_interfaces::srv::SetInitialPose::Request> request,
        std::shared_ptr<initial_pose_interfaces::srv::SetInitialPose::Response> response)
    {
        std::lock_guard<std::mutex> lock(mtx);  // Lock for the entire callback
        
        auto pose_msg = std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>(request->pose);
        initialposeHandler(pose_msg);

        if (systemInitialize())
        {
            response->success = true;
            response->message = "Initial pose set and initialization successful";
        }
        else
        {
            response->success = false;
            response->message = "Initial pose set but initialization failed";
        }
    }
};

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader
RCLCPP_COMPONENTS_REGISTER_NODE(mapOptimization)

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();

    // loopthread.join();
    // visualizeMapThread.join();

    return 0;
}
