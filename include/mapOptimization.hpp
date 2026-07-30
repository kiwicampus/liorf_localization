#pragma once

// clang-format off
// utility.h must be included before any gtsam header: it pulls in the
// system Eigen (via rclcpp/PCL/tf2_eigen) that GTSAM's own headers then
// reuse via include guards. Reversing this order lets GTSAM's Eigen-version
// static_assert see a different EIGEN_MAJOR_VERSION than GTSAM_EIGEN_VERSION_MAJOR
// expects and fails to compile with "GTSAM was built against a different
// version of Eigen". Keep this block unsorted so clang-format's
// IncludeBlocks: Preserve doesn't merge it with the gtsam block below.
#include "utility.h"
#include "nanoflann_pcl.h"
#include "liorf_localization/msg/cloud_info.hpp"
#include "liorf_localization/msg/localization_info.hpp"
#include "liorf_localization/srv/save_map.hpp"
#include "std_msgs/msg/empty.hpp"
// clang-format on

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "initial_pose_interfaces/srv/set_initial_pose.hpp"

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;                     // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                                      float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

// Declaration-only: method bodies live in mapOptmization.cpp / mapOptimizationFactors.cpp.
// A single mapOptmization.cpp TU compiling this whole class inline peaked at ~4.2GB RSS
// (PCL+GTSAM+Eigen template instantiation, unaffected by -O level) and was getting
// OOM-killed on the Balena builder even fully serialized. Splitting method definitions
// across two TUs keeps each compile's peak lower since neither TU instantiates the full
// set of factor-graph/ICP template combinations at once.
class mapOptimization : public ParamServer
{
   public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2* isam;
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

    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;    // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;  // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriSurfVec;  // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    nanoflann::KdTreeFLANN<PointType> kdtreeSurfFromMap;

    nanoflann::KdTreeFLANN<PointType> kdtreeSurroundingKeyPoses;
    nanoflann::KdTreeFLANN<PointType> kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterLocalMapSurf;
    pcl::VoxelGrid<PointType>
        downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

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
    map<int, int> loopIndexContainer;  // from new to old
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

    mapOptimization(const rclcpp::NodeOptions& options);

    rcl_interfaces::msg::SetParametersResult on_parameters_set_callback(
        const std::vector<rclcpp::Parameter>& parameters);

    void allocateMemory();

    // add by yjz_lucky_boy
    void loadGlobalMap();

    void processReceivedMap(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void processLoadedMap(pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap);

    // add by yjz_lucky_boy
    void initialposeHandler(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msgIn);

    void laserCloudInfoHandler(const liorf_localization::msg::CloudInfo::SharedPtr msgIn);

    bool systemInitialize();

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg);

    void pointAssociateToMap(PointType const* const pi, PointType* const po);

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                        PointTypePose* transformIn);

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);

    gtsam::Pose3 trans2gtsamPose(float transformIn[]);

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);

    Eigen::Affine3f trans2Affine3f(float transformIn[]);

    PointTypePose trans2PointTypePose(float transformIn[]);

    bool saveMapService(const std::shared_ptr<liorf_localization::srv::SaveMap::Request> req,
                        std::shared_ptr<liorf_localization::srv::SaveMap::Response> res);

    void visualizeGlobalMapThread();

    void publishGlobalMap();

    void updateInitialGuess();

    void downsampleCurrentScan();

    void adjustForRotation();

    void updatePointAssociateToMap();

    void surfOptimization();

    void combineOptimizationCoeffs();

    bool LMOptimization(int iterCount);

    // Optimizer-internal match quality for match_quality_monitor.
    void updateMatchQualityMetrics(bool optimization_converged);

    // Post-pose scan-to-map fitness for map_fitness_monitor and pose_jump_monitor.
    void updateMapFitnessMetrics();

    // <!-- liorf_localization_yjz_lucky_boy -->
    bool scan2MapOptimization();

    void transformUpdate();

    float constraintTransformation(float value, float limit);

    bool saveFrame();

    void addOdomFactor();

    void addGPSFactor();

    void saveKeyFramesAndFactor();

    void correctPoses();

    void updatePath(const PointTypePose& pose_in);

    void publishOdometry();

    void publishFrames();

    template <typename T>
    bool publishThrottle(const typename rclcpp::Publisher<T>::SharedPtr& pub, const T& msg, double min_interval)
    {
        static std::unordered_map<std::string, rclcpp::Time> last_pub_time;
        auto now = this->now();
        auto topic = pub->get_topic_name();
        if (last_pub_time.find(topic) == last_pub_time.end())
        {
            last_pub_time[topic] = now - rclcpp::Duration::from_seconds(min_interval);
        }
        if ((now - last_pub_time[topic]).seconds() >= min_interval)
        {
            pub->publish(msg);
            last_pub_time[topic] = now;
            return true;
        }
        return false;
    }

    void setInitialPoseCallback(const std::shared_ptr<initial_pose_interfaces::srv::SetInitialPose::Request> request,
                                std::shared_ptr<initial_pose_interfaces::srv::SetInitialPose::Response> response);
};
