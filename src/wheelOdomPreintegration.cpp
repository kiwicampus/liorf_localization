#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubWheelOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubWheelPath;

    Eigen::Affine3f wheelOdomAffineFront;
    Eigen::Affine3f lidarOdomAffine;
    deque<nav_msgs::msg::Odometry> wheelOdomQueue;

    shared_ptr<tf2_ros::Buffer> tfBuffer;
    shared_ptr<tf2_ros::TransformListener> tfListener;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfMap2Odom;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfOdom2BaseLink;
    tf2::Stamped<tf2::Transform> lidar2Baselink;
    double lidarOdomTime = -1;

    TransformFusion(const rclcpp::NodeOptions& options) : ParamServer("transform_fusion", options)
    {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        tfMap2Odom = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        tfOdom2BaseLink = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tf2::fromMsg(tfBuffer->lookupTransform(
                    lidarFrame, baselinkFrame, rclcpp::Time(0)), lidar2Baselink);
            }
            catch (tf2::TransformException ex)
            {
                RCLCPP_ERROR(get_logger(), "%s", ex.what());
            }
        }

        subWheelOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", QosPolicy(history_policy, reliability_policy),
            bind(&TransformFusion::wheelOdometryHandler, this, placeholders::_1));
        
        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>("liorf_localization/mapping/odometry", QosPolicy(history_policy, reliability_policy), 
                    std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1));

        pubWheelOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, QosPolicy(history_policy, reliability_policy));
        pubWheelPath = create_publisher<nav_msgs::msg::Path>("/wheel_odometry/path", QosPolicy(history_policy, reliability_policy));
    }

    Eigen::Affine3f odom2affine(nav_msgs::msg::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf2::Quaternion orientation(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = ROS_TIME(odomMsg->header.stamp);
    }

    void wheelOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        // static tf
        tf2::Quaternion quat_tf;
        rclcpp::Time t(static_cast<uint32_t>(lidarOdomTime * 1e9));
        tf2::TimePoint time_point = tf2_ros::fromRclcpp(t);
        std::lock_guard<std::mutex> lock(mtx);

        wheelOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current WheelOdom stamp)
        if (lidarOdomTime == -1)
            return;
        while (!wheelOdomQueue.empty())
        {
            if (ROS_TIME(wheelOdomQueue.front().header.stamp) <= lidarOdomTime)
                wheelOdomQueue.pop_front();
            else
                break;
        }

        Eigen::Affine3f wheelOdomAffineFront = odom2affine(wheelOdomQueue.front());
        Eigen::Affine3f wheelOdomAffineBack = odom2affine(wheelOdomQueue.back());
        Eigen::Affine3f wheelOdomAffineIncre = wheelOdomAffineFront.inverse() * wheelOdomAffineBack;
        Eigen::Affine3f wheelOdomAffineLast = lidarOdomAffine * wheelOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(wheelOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        nav_msgs::msg::Odometry laserOdometry = wheelOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        quat_tf.setRPY(roll, pitch, yaw);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometry.pose.pose.orientation = quat_msg;
        pubWheelOdometry->publish(laserOdometry);

        // publish tf
        tf2::Transform tCur(tf2::Quaternion(laserOdometry.pose.pose.orientation.x, laserOdometry.pose.pose.orientation.y, laserOdometry.pose.pose.orientation.z, laserOdometry.pose.pose.orientation.w), 
                                tf2::Vector3(laserOdometry.pose.pose.position.x, laserOdometry.pose.pose.position.y, laserOdometry.pose.pose.position.z));
        if(lidarFrame != baselinkFrame)
            tCur *= lidar2Baselink;

        // tf2::Stamped<tf2::Transform> temp_odom_to_base(tCur, time_point, mapFrame);
        // geometry_msgs::msg::TransformStamped trans_odom_to_base_link;
        // tf2::convert(temp_odom_to_base, trans_odom_to_base_link);
        // trans_odom_to_base_link.child_frame_id = baselinkFrame;
        // tfOdom2BaseLink->sendTransform(trans_odom_to_base_link);

        // publish WheelOdom path
        static nav_msgs::msg::Path wheelOdomPath;
        static double last_path_time = -1;
        double wheelOdomTime = ROS_TIME(wheelOdomQueue.back().header.stamp);
        if (wheelOdomTime - last_path_time > 0.1)
        {
            last_path_time = wheelOdomTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = wheelOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = mapFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            wheelOdomPath.poses.push_back(pose_stamped);
            while(!wheelOdomPath.poses.empty() && ROS_TIME(wheelOdomPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                wheelOdomPath.poses.erase(wheelOdomPath.poses.begin());
            if (pubWheelPath->get_subscription_count() != 0)
            {
                wheelOdomPath.header.stamp = wheelOdomQueue.back().header.stamp;
                wheelOdomPath.header.frame_id = mapFrame;
                pubWheelPath->publish(wheelOdomPath);
            }
        }
    }
};

class WheelOdometryPreintegration : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subIncrementalOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdometry;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;

    std::deque<nav_msgs::msg::Odometry> wheelQueOpt;
    std::deque<nav_msgs::msg::Odometry> wheelQue;

    bool systemInitialized = false;
    int key = 1;

    const double delta_t = 0;

    bool doneFirstOpt = false;
    float wheelOdomRate = 30.0;

    double lastUpdateTime = 0;

    // Wheel to lidar: 0.0, 0.0, 0.6 (x, y, z) + 0., 0.12467473, 0., 0.99219767 (q.x, q.y, q.z, q.w)
    gtsam::Pose3 wheel2Lidar = gtsam::Pose3(gtsam::Rot3::Quaternion(0.99219767, 0.0, 0.12467473, 0.0), gtsam::Point3(0.0, 0.0, 0.6));
    // Lidar to Wheel:[ 0.14844238  0.         -0.58134745] (x,y,z) + [ 0.         -0.12467473  0.          0.99219767] (q.x, q.y, q.z, q.w)
    gtsam::Pose3 lidar2Wheel = gtsam::Pose3(gtsam::Rot3::Quaternion(0.99219767, 0.0, -0.12467473, 0.0), gtsam::Point3(0.14844238, 0.0, -0.58134745));

    WheelOdometryPreintegration(const rclcpp::NodeOptions& options) :
            ParamServer("wheel_odometry_preintegration", options)
    {
        subWheelOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "/wheel_odometry/global_odometry", 10,
            std::bind(&WheelOdometryPreintegration::wheelOdometryHandler, this, std::placeholders::_1));

        // subIncrementalOdometry = create_subscription<nav_msgs::msg::Odometry>(
        //     "liorf_localization/mapping/odometry_incremental", 10,
        //     std::bind(&WheelOdometryPreintegration::incrementalOdometryHandler, this, std::placeholders::_1));

        pubOdometry = create_publisher<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", 10);

        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    // void integrateAndOptimize(const gtsam::Pose3& pose, const rclcpp::Time& timestamp)
    // {
         // Raw odometry updates are not added to the factor graph!
    //     // Add pose to the graph
    //     gtsam::PriorFactor<gtsam::Pose3> poseFactor(X(key), pose, correctionNoise);
    //     graphFactors.add(poseFactor);

    //     // Update and optimize the graph
    //     graphValues.insert(X(key++), pose);
    //     optimizer.update(graphFactors, graphValues);
    //     optimizer.update();
    //     graphFactors.resize(0);
    //     graphValues.clear();
    // }

    void applyCorrection(const gtsam::Pose3& correctedPose, const bool degenerate, const double timestamp)
    {
        // Reset the graph at regular intervals or when needed
        if (key == 100) {

            // Reinitialize the graph with the last known good state
            // Assume that `prevPose_`, `prevVel_`, and other necessary state variables are correctly maintained
            gtsam::noiseModel::Diagonal::shared_ptr updatedPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1).finished());
            gtsam::noiseModel::Isotropic::shared_ptr updatedVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4);
            resetOptimization();
            
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);

            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);

            // Reinsert the initial state into the graph
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);

            // Optionally, reset other state variables as needed

            // Optimize the graph with the initial state
            optimizer.update(graphFactors, graphValues);
            // optimizer.update();
            graphFactors.resize(0);
            graphValues.clear();

            key = 1; // Reset the key index
        }

        // Add corrected pose to the graph
        // initial pose
        gtsam::Pose3 curPose = correctedPose.compose(lidar2Wheel);
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(priorPose);

        // Insert the corrected pose and optimize
        graphValues.insert(X(key), prevState_.pose());
        graphValues.insert(V(key), prevState_.velocity());
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();

        // Update the previous pose and orientation to the corrected pose
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        // prevPose_ = correctedPose.translation();
        // prevTheta_ = correctedPose.rotation().yaw();
        ++key;
        doneFirstOpt = true;
    }

    void wheelOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // if (doneFirstOpt == false)
        //     return;

        // Calculate time difference
        double currentTime = ROS_TIME(odomMsg->header.stamp);
        if (lastUpdateTime == 0) { // Initialize if it's the first message
            lastUpdateTime = currentTime;
            return;
        }
        double dt = currentTime - lastUpdateTime;

        // Simple integration to update pose
        double dx = odomMsg->twist.twist.linear.x * dt;
        double dy = odomMsg->twist.twist.linear.y * dt;
        double dtheta = odomMsg->twist.twist.angular.z * dt;

        // Update pose estimates
        double prevTheta = prevPose_.rotation().yaw();
        double newX = prevPose_.x() + (dx * cos(prevTheta) - dy * sin(prevTheta));
        double newY = prevPose_.y() + (dx * sin(prevTheta) + dy * cos(prevTheta));
        double newTheta = prevTheta + dtheta;
        double prev_z = prevPose_.z();

        // Create a new pose
        gtsam::Pose3 newPose(gtsam::Rot3::Rz(newTheta), gtsam::Point3(newX, newY, prev_z));

        // Update factor graph
        // integrateAndOptimize(newPose, currentTime);

        // Save for next update
        // prevPose_ = gtsam::Point3(newX, newY, 0);
        prevPose_ = newPose;
        prevVel_ = gtsam::Vector3(odomMsg->twist.twist.linear.x, odomMsg->twist.twist.linear.y, 0);
        prevState_ = gtsam::NavState(newPose, prevVel_);
        // prevTheta_ = newTheta;
        lastUpdateTime = currentTime;

        // Publish odometry
        publishTransformedOdometry(odomMsg->header.stamp);

    }

    void incrementalOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // Extract and process the corrected pose in a similar manner
        gtsam::Pose3 correctedPose(gtsam::Rot3::Quaternion(odomMsg->pose.pose.orientation.w,
                                                        odomMsg->pose.pose.orientation.x,
                                                        odomMsg->pose.pose.orientation.y,
                                                        odomMsg->pose.pose.orientation.z),
                                gtsam::Point3(odomMsg->pose.pose.position.x,
                                                odomMsg->pose.pose.position.y,
                                                odomMsg->pose.pose.position.z));
        double curr_time = ROS_TIME(odomMsg->header.stamp);
        if (!systemInitialized) {
            initializeSystemWithCorrectedPose(correctedPose, curr_time);
        } else {
            bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
            applyCorrection(correctedPose, degenerate, curr_time);
        }
    }

    void initializeSystemWithCorrectedPose(const gtsam::Pose3& correctedPose, const double currentTime)
    {

        // Reset any existing graph or values
        resetOptimization();

        // Initialize the system with the corrected pose
        prevPose_ = correctedPose;
        // prevTheta_ = correctedPose.rotation().yaw(); // Extracting the yaw component for 2D orientation
        lastUpdateTime = currentTime;

        // Initialize factor graph with the corrected pose
        gtsam::noiseModel::Diagonal::shared_ptr initialPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1).finished());
        gtsam::PriorFactor<gtsam::Pose3> initialPoseFactor(X(0), correctedPose, initialPoseNoise);
        graphFactors.add(initialPoseFactor);

        // Add initial velocity assuming it starts stationary
        gtsam::Vector3 initialVelocity(0, 0, 0);
        gtsam::noiseModel::Isotropic::shared_ptr velocityNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4);
        gtsam::PriorFactor<gtsam::Vector3> velocityFactor(V(0), initialVelocity, velocityNoise);
        graphFactors.add(velocityFactor);

        // Insert initial values into the graph
        graphValues.insert(X(0), correctedPose);
        graphValues.insert(V(0), initialVelocity);

        // Run initial optimization
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();

        // Mark the system as initialized
        systemInitialized = true;
        key = 1;
    }

    void publishTransformedOdometry(const rclcpp::Time& timestamp)
    {
        gtsam::Pose3 pose = prevState_.pose();
        gtsam::Pose3 lidarPose = pose.compose(wheel2Lidar);
        // velocity
        gtsam::Vector3 velocity = prevState_.velocity();
        nav_msgs::msg::Odometry odometry;
        odometry.header.stamp = timestamp;
        odometry.header.frame_id = mapFrame;
        odometry.child_frame_id = "odom_imu";

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        auto quat = lidarPose.rotation().toQuaternion();
        odometry.pose.pose.orientation.x = quat.x();
        odometry.pose.pose.orientation.y = quat.y();
        odometry.pose.pose.orientation.z = quat.z();
        odometry.pose.pose.orientation.w = quat.w();

        // odometry.twist.twist.linear.x = velocity.x();
        // odometry.twist.twist.linear.y = velocity.y();
        // odometry.twist.twist.linear.z = velocity.z();

        pubOdometry->publish(odometry);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<WheelOdometryPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
