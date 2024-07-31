#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/GPSFactor.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)


class WheelOdometryPreintegration : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdometry;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;

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

        pubOdometry = create_publisher<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", 10);

    }

    void wheelOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

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

        // Save for next update
        prevPose_ = newPose;
        prevVel_ = gtsam::Vector3(odomMsg->twist.twist.linear.x, odomMsg->twist.twist.linear.y, 0);
        prevState_ = gtsam::NavState(newPose, prevVel_);
        lastUpdateTime = currentTime;

        // Publish odometry
        publishTransformedOdometry(odomMsg->header.stamp);

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
    e.add_node(ImuP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
