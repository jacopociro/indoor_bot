/*
TODO:
trovare il modo di parametrizzare la velocità lineare tra 0 e max_vel in base alla lunghezza del vettore risultante, in modo da rallentare quando il robot si avvicina alla meta o quando ci sono molti ostacoli intorno
sistema help in modo che chieda solo le risorse che servono (modifica il service)
grafico in tempo reale anche della batteria 
salvataggio del tempo di completamento della missione (sim time )
/master_discovery/changes
/master_discovery/linkstats
/rosbot_1/camera/rgb/image_raw/compressed
/rosbot_1/cmd_vel
/rosbot_1/odom
/rosbot_1/scan
/rosbot_1/battery
/rosbot_2/camera/rgb/image_raw/compressed
/rosbot_2/cmd_vel
/rosbot_2/odom
/rosbot_2/scan
/rosbot_2/battery
/rosout
/rosout_agg
/tf
/tf_static
modifica in modo che questi siano i ns e i topic
*/





#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/BatteryState.h>
#include <yaml-cpp/yaml.h>
#include <XmlRpcValue.h>
#include "indoor_bot/resourcesrv.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <deque>
#include <fstream>
#include <iomanip>
#include <algorithm>
// ========================
// Struttura Waypoint
// ========================
struct Point2D {
    double x, y, yaw;

    bool operator==(const Point2D& other) const {
        const double epsilon = 1e-6;
        return std::fabs(x - other.x) < epsilon &&
               std::fabs(y - other.y) < epsilon &&
               std::fabs(yaw - other.yaw) < epsilon;
    }
};
struct Pose {
    double x, y, z, yaw;
};
class IndoorBot
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher velocity_pub_;
    ros::Publisher pose_pub;
    ros::Subscriber scan_sub_;
    ros::Subscriber odom_sub_;
    std::vector<ros::Subscriber> odom_subs_;
    ros::Subscriber battery_sub_;
    ros::ServiceClient resource_client_;
    std::vector<Pose> waypoints_;

    std::ofstream pos_file_;
    std::ofstream priority_file_;
    std::ofstream direction_file_;
    std::ofstream battery_file_;

    double radius = 0.1;      // real: , sim:
    float obs_gvalue = 0.1f; // 0.1, 0.15,  Gaussian value for obstacles
    float drones_gvalue = 0.4f; // 0.4, 0.4,  Gaussian value for other drones
    float docks_gvalue = 6.0f; // 6.0, 6.0,  Gaussian value for charging stations
    float wp_gvalue = 3.0f; // 3.0, 4.6,  Gaussian value for waypoints
    float split = 0.95;
    double max_vel = 0.05;//0.05;
    const size_t MAX_SIZE = 1440;
    bool returning = false;
    bool help0 = false;
    bool help1 = false;
    bool help2 = false;
    bool help3 = false;
    bool help4 = false;
    std::vector<double> Memory;
    std::vector<double> Memory_temp;
    std::vector<double> Priority;
    std::vector<double> Mem_leader;
    std::vector<bool> discovered_;
    // Posizione robot
    std::vector<Point2D> robot;
    bool odom_received_;
    std::deque<Point2D> obs_positions;
    std::vector<Point2D> sensing_points;
    std::vector<Point2D> resulting_vectors;

    XmlRpc::XmlRpcValue stations_param;
    XmlRpc::XmlRpcValue init_pos_param;
    std::vector<Pose> charging_stations;
    std::vector<Pose>  init_pos;
    int n_UGV;
    std::string name;
    int id;


    std::vector<std::string> ugv_names;
    std::vector<int> ugv_idx;
    // ========================
    // Battery simulation
    // ========================
    bool simulation_mode_ = false;

    ros::Publisher battery_pub_;

    double battery_soc_ = 1.0;          // [0,1]
    double battery_capacity_ah_ = 2.5; // 3x18650 in serie
    double battery_current_ = 0.0;

    // Parametri robot ROSbot 2.0
    double robot_mass_ = 2.84;          // kg
    double wheel_radius_ = 0.0425;      // m
    double rolling_coeff_ = 0.02;
    double motor_kt_ = 0.017;           // Nm/A
    double drivetrain_efficiency_ = 0.85;
    double aux_current_ = 0.8;          // A

    // Thevenin ECM
    double R0_ = 0.15;                  // ohm
    double R1_ = 0.03;                  // ohm
    double C1_ = 2000.0;                // F
    double V_rc_ = 0.0;

    // Stato precedente per accelerazione
    double prev_linear_velocity_ = 0.0;
    double prev_update_time_ = 0.0;

    float volt_min = 9.0f; // Minimum voltage for battery
    float volt_max = 12.6f; // Maximum voltage for battery
    float battery_voltage_ = volt_max; // Current battery voltage
    float battery_percentage_ = 100.0f; // Current battery percentage

    bool mission_timer_started_ = false;
    bool mission_timer_stopped_ = false;
    bool charging_timer_stopped_ = false;

    double mission_start_time_ = 0.0;
    double mission_completion_time_ = 0.0;
    double charging_completion_time_ = 0.0;
    double time = 0.0;
    std::ofstream mission_time_file_;
    bool mission_time_saved_ = false;

    std::deque<float> voltage_buffer_;
    int window_size_ = 10;  // numero di campioni per la media

public:
    IndoorBot(const std::string& waypoint_file)
    : nh_(),
      pnh_("~"),
      odom_received_(false)
{
    // PARAMETERS
    pnh_.param("n_UGV", n_UGV, 1);
    pnh_.param("robot_name", name, std::string("robot"));
    pnh_.param("robot_id", id, 0);
    pnh_.param("simulation_mode", simulation_mode_, false);

    pnh_.param("battery_capacity_ah", battery_capacity_ah_, 2.5);
    pnh_.param("battery_r0", R0_, 0.15);
    pnh_.param("battery_r1", R1_, 0.03);
    pnh_.param("battery_c1", C1_, 2000.0);
    pnh_.param("battery_aux_current", aux_current_, 0.8);

    if (!nh_.getParam("/init_pos", init_pos_param))
    {
        ROS_ERROR("Failed to get init_pos parameter");
    }
    else if (init_pos_param.getType() == XmlRpc::XmlRpcValue::TypeArray)
    {   
        ROS_INFO("Got init_pos parameter with %d entries", init_pos_param.size());
        for (int i = 0; i < init_pos_param.size(); ++i)
        {
            XmlRpc::XmlRpcValue pos = init_pos_param[i];
            
            Pose p;
            p.x = static_cast<double>(pos["x"]);
            p.y = static_cast<double>(pos["y"]);
            p.z = static_cast<double>(pos["z"]);
            p.yaw = static_cast<double>(pos["yaw"]);

            init_pos.push_back(p);
        }
    }

    if (!nh_.getParam("/charging_station", stations_param))
    {
        ROS_ERROR("Failed to get charging_station parameter");
    }
    else if (stations_param.getType() == XmlRpc::XmlRpcValue::TypeArray)
    {
        for (int i = 0; i < stations_param.size(); ++i)
        {
            XmlRpc::XmlRpcValue station = stations_param[i];

            Pose p;
            p.x = static_cast<double>(station["x"]);
            p.y = static_cast<double>(station["y"]);
            p.z = static_cast<double>(station["z"]);
            p.yaw = static_cast<double>(station["yaw"]);

            charging_stations.push_back(p);
        }
    }

    for (size_t i = 0; i < charging_stations.size(); ++i)
    {
        ROS_INFO("Station %zu: x=%.2f y=%.2f z=%.2f yaw=%.2f",
                 i,
                 charging_stations[i].x,
                 charging_stations[i].y,
                 charging_stations[i].z,
                 charging_stations[i].yaw);
    }

    nh_.getParam("/UGV_names", ugv_names);
    nh_.getParam("/UGV_idx", ugv_idx);

    robot.resize(ugv_names.size());

    for (size_t i = 0; i < ugv_names.size(); i++)
    {
        ROS_INFO("Robot %zu -> name: %s index: %d",
                 i,
                 ugv_names[i].c_str(),
                 ugv_idx[i]);

        robot[i] = Point2D{0.0, 0.0, 0.0};
    }
    if (ugv_names.size() != ugv_idx.size())
    {
        ROS_ERROR("UGV_names and UGV_idx size mismatch!");
    }
    // PUBLISHERS
    velocity_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    pose_pub = nh_.advertise<geometry_msgs::Pose>("goal", 10);

    // SUBSCRIBERS
    scan_sub_ = nh_.subscribe("scan", 1, &IndoorBot::scanCallback, this);

    odom_subs_.reserve(ugv_names.size());

    for (size_t i = 0; i < ugv_names.size(); i++)
    {
        std::string topic = "/" + ugv_names[i] + "/odom";
        ROS_INFO("Subscribing to %s", topic.c_str());
        odom_subs_.push_back(
            nh_.subscribe<nav_msgs::Odometry>(
                topic,
                1,
                [this, i](const nav_msgs::Odometry::ConstPtr& msg)
                {
                    odomCallback(msg, i);
                }
            )
        );
    }
     if (simulation_mode_)
    {
        battery_pub_ = nh_.advertise<sensor_msgs::BatteryState>("battery", 10);

        battery_voltage_ = volt_max;
        battery_percentage_ = 100.0f;
        battery_soc_ = 1.0;

        ROS_INFO("Battery simulation enabled");
    }
    else
    {
        battery_sub_ = nh_.subscribe("battery", 10, &IndoorBot::batteryCallback, this);

        ROS_INFO("Battery topic subscription enabled");
    }
    // WAYPOINTS
    waypoints_ = readWaypoints(waypoint_file);

    for (const auto& wp : waypoints_)
    {
        Memory.push_back(0.0);
        Memory_temp.push_back(0.0);
        Mem_leader.push_back(0.0);
        Priority.push_back(0.0);
        discovered_.push_back(true);
    }
    discovered_[4] = false; // Waypoint 5 is initially undiscovered
    Priority.push_back(0.0); // Battery priority
    if (simulation_mode_){
        ros::service::waitForService("/gazebo/get_world_properties");

        ROS_INFO("Gazebo is ready!");
    }
    resource_client_ = nh_.serviceClient<indoor_bot::resourcesrv>("resource_service");
    std::string base_name = "/root/catkin_ws/src/indoor_bot/indoor_bot/data/" + name + "_" + std::to_string(id);

    pos_file_.open(base_name + "_position.csv");
    priority_file_.open(base_name + "_priority.csv");
    direction_file_.open(base_name + "_direction.csv");
    battery_file_.open(base_name + "_battery.csv");
    mission_time_file_.open(base_name + "_mission_times.csv");
    mission_time_file_ << "mission_completion_time,charging_completion_time\n";

    // Header CSV
    pos_file_ << "time,x,y,yaw\n";

    priority_file_ << "time";
    for (size_t i = 0; i < Priority.size(); i++)
        priority_file_ << ",p" << i;
    priority_file_ << "\n";

    direction_file_ << "time,dir_x,dir_y,yaw\n";

    battery_file_ << "time,voltage,percentage\n";
    
    mission_timer_started_ = true;
}

    // ========================
    // Lettura YAML
    // ========================
    std::vector<Pose> readWaypoints(const std::string& filename)
    {
        std::vector<Pose> waypoints;
        int i;
        try {
            YAML::Node config = YAML::LoadFile(filename);

            if (config["wp"]) {
                i = 0;
                for (const auto& wp : config["wp"]) {
                    Pose waypoint;
                    waypoint.x   = wp["x"].as<double>();
                    waypoint.y   = wp["y"].as<double>();
                    waypoint.z   = wp["z"].as<double>();
                    waypoint.yaw = wp["yaw"].as<double>();
                    waypoints.push_back(waypoint);
                    //ROS_INFO("Waypoint %i: x=%.2f y=%.2f z=%.2f yaw=%.2f", i, waypoint.x, waypoint.y, waypoint.z, waypoint.yaw);
                    i = i + 1;
                }
            }
        }
        catch (const YAML::Exception& e) {
            ROS_ERROR("Errore lettura YAML: %s", e.what());
        }

        return waypoints;
    }

    // ========================
    // Callback ODOM
    // ========================
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg, int id_)
    {
        // rotated_wp.x = cos_angle * wp.x + sin_angle * wp.z;
        // rotated_wp.y = wp.y; // Y remains unchanged
        // rotated_wp.z = -sin_angle * wp.x + cos_angle * wp.z;
        // rotated_wp.yaw = wp.yaw; // Yaw remains unchanged
        double angle = M_PI;
        robot[id_].x = msg->pose.pose.position.x + init_pos[id_].x;
        robot[id_].y = msg->pose.pose.position.y + init_pos[id_].y;
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;

        // Conversione da quaternione a yaw, pitch, roll
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        robot[id_].yaw = std::atan2(siny_cosp, cosy_cosp);
        odom_received_ = true;
        //ROS_INFO("Current robot position: x=%.2f, y=%.2f, yaw=%.2f", robot[id_].x, robot[id_].y, robot[id_].yaw);
    }

    // ========================
    // Callback SCAN
    // ========================
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        obs_positions.clear();
        float angle = 0.0;
        if (!odom_received_)
            return;

        double min_distance = 1.0;
        Point2D obs_;
        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            
            float range = msg->ranges[i];

            if (std::isinf(range) || std::isnan(range))
                continue;
            
            angle = msg->angle_min + i * msg->angle_increment;
            if (simulation_mode_ == false){
                // obs.x = - obs.x;
                // obs.y = - obs.y;
                angle = angle + M_PI; // Correzione per orientamento differente in real vs sim
            }
            // Coordinate ostacolo (assumendo stesso frame)
            Point2D obs;
            obs.x = robot[id].x + range * std::cos(angle + robot[id].yaw);
            obs.y = robot[id].y + range * std::sin(angle + robot[id].yaw);

            double dist = distance2D(robot[id].x, robot[id].y, obs.x, obs.y);
            
            if (dist < min_distance)
            {
                // if (std::find(obs_positions.begin(), obs_positions.end(), obs) == obs_positions.end())
                {
                    // if (obs_positions.size() > MAX_SIZE) {
                    //     obs_positions.pop_front();
                    // }
                    obs_positions.push_back(obs);
                    //ROS_INFO("Scan %zu: range=%.2f angle=%.2f -> obs_x=%.2f obs_y=%.2f dist=%.2f", i, range, angle, obs.x, obs.y, dist);
                }
                
                // obs_positions.push_back(obs);
                // //min_distance = dist;
                // obs_.x = obs.x;
                // obs_.y = obs.y;
            }
            

        }
        //ROS_INFO("obs pos size %i", obs_positions.size());
        // obs_positions.push_back(obs_);

        
    }
    // ========================
    // Callback BatteryState
    // ========================
    void batteryCallback(const sensor_msgs::BatteryState::ConstPtr& msg)
    {
        float volt_raw = msg->voltage;

        // Aggiungi il nuovo valore al buffer
        voltage_buffer_.push_back(volt_raw);

        // Mantieni la dimensione fissa
        if (voltage_buffer_.size() > window_size_) {
            voltage_buffer_.pop_front();
        }

        // Calcola la media
        float sum = 0.0f;
        for (float v : voltage_buffer_) {
            sum += v;
        }
        float volt_filtered = sum / voltage_buffer_.size();

        // ROS_INFO("Voltage (raw): %.2f V", volt_raw);
        // ROS_INFO("Voltage (filtered): %.2f V", volt_filtered);
        // ROS_INFO("Battery percentage: %.1f%%", msg->percentage);

        float percentage = msg->percentage;

        if (!simulation_mode_) {
            Priority[Priority.size() - 1] =
                -0.1 + (volt_filtered - volt_max) / (volt_min - volt_max) * 1.1;
        } else {
            Priority[Priority.size() - 1] =
                -0.1 + ((100.0f - battery_percentage_) / 100.0f) * 1.1;
        }
        if (returning == true){
            Priority[Priority.size() - 1] = 1.5f; // Massima priorità per il ritorno alla stazione di ricarica
        }
        battery_voltage_ = volt_filtered;
        battery_percentage_ = percentage;
    }
    // ========================
    // Distanza 2D
    // ========================
    double distance2D(double x1, double y1, double x2, double y2)
    {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return std::sqrt(dx * dx + dy * dy);
    }
    // ========================
    // Genera punti intorno al robot
    // ========================
    std::vector<Point2D> generateSurroundingPoints(double radius)
    {
        std::vector<Point2D> points;
        const int num_points = 8;
        const double angle_increment = 2 * M_PI / num_points;

        for (int i = 0; i < num_points; ++i)
        {
            double angle = i * angle_increment;
            Point2D point;
            point.x = robot[id].x + radius * std::cos(angle);
            point.y = robot[id].y + radius * std::sin(angle);
            points.push_back(point);
        }

        return points;
    }

    double pidYawControl(double error, double kp, double ki, double kd)
    {
        static double previous_error = 0.0;
        static double integral = 0.0;

        double derivative = error - previous_error;
        integral += error;

        double yaw_speed = kp * error + ki * integral + kd * derivative;

        previous_error = error;

        return yaw_speed;
    }

    float biological_calculation(int counter, float perc_value, bool uptake)
    {
        float TF_Imax = -6.992f * std::pow(10.0f, -3.0f) * Memory[counter] 
                        + 3.672f * std::pow(10.0f, -6.0f);

        float TF_Kmax = -8.4f * std::pow(10.0f, -4.0f) * Memory[counter] 
                        + 61.0f;

        float P = 1.1f * std::pow(TF_Imax, 2.0f) 
                / std::pow(36.72f * std::pow(10.0f, -7.0f), 2.0f) 
                - 0.1f;

        float I = TF_Imax * perc_value / (TF_Kmax + perc_value)* 100.0f;

        if (uptake)
        {
            Memory_temp[counter] = I * split;
            Mem_leader[counter]  = (I - I * split);

        }
        else        {
            Memory_temp[counter] = 0.0; //I * split;
            Mem_leader[counter]  = 0.0; //I - I * split;
        }
        return P;
    }
    double gaussian(double x, double stddev)
    {
        if (stddev <= 0.0)
            return 0.0;

        constexpr double PI = 3.14159265358979323846;

        const double variance = stddev * stddev;
        const double g = std::exp(-(x * x) / (2.0 * variance))
                        / (stddev * std::sqrt(2.0 * PI));

        return g * 1.01;
    }
        // ========================
    // Rotazione Waypoints
    // ========================
    std::vector<Pose> rotateWaypoints(const std::vector<Pose>& waypoints, double angle)
    {
        std::vector<Pose> rotated_waypoints;

        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);

        for (const auto& wp : waypoints)
        {
            Pose rotated_wp;
            rotated_wp.x = cos_angle * wp.x + sin_angle * wp.z;
            rotated_wp.y = wp.y; // Y remains unchanged
            rotated_wp.z = -sin_angle * wp.x + cos_angle * wp.z;
            rotated_wp.yaw = wp.yaw; // Yaw remains unchanged
            rotated_waypoints.push_back(rotated_wp);
        }

        return rotated_waypoints;
    }

    void sendRequest()
    {
        indoor_bot::resourcesrv srv;
        
        srv.request.request.resource0 = Mem_leader[0];
        srv.request.request.resource1 = Mem_leader[1];
        srv.request.request.resource2 = Mem_leader[2];
        srv.request.request.resource3 = Mem_leader[3];
        srv.request.request.resource4 = Mem_leader[4];
        // Da modificare per inspection
        // if (help0 == true){
        //     srv.request.request.resource0 = Mem_leader[0] + Memory[0]/3;

        //     Memory[0] = Memory[0] - Memory[0]/3;
        //     ROS_INFO("Help needed: %s", help1 ? "Yes" : "No");
        // }
        // if (help1 == true){
        //     srv.request.request.resource1 = Mem_leader[1] + Memory[1]/3;

        //     Memory[1] = Memory[1] - Memory[1]/3;
        //     ROS_INFO("Help needed: %s", help1 ? "Yes" : "No");
        // }
        // if (help2 == true){
        //     srv.request.request.resource2 = Mem_leader[2] + Memory[2]/3;    
        //     Memory[2] = Memory[2] - Memory[2]/3;
        //     ROS_INFO("Help needed: %s", help2 ? "Yes" : "No ");
        // }
        // if (help3 == true){
        //     srv.request.request.resource3 = Mem_leader[3] + Memory[3]/3 ;   
        //     Memory[3] = Memory[3] - Memory[3]/3;
        //     ROS_INFO("Help needed: %s", help3 ? "Yes" : "No");
        // }
        // if (help4 == true){
        //     srv.request.request.resource4 = Mem_leader[4] + Memory[4]/3;    
        //     Memory[4] = Memory[4] - Memory[4]/3;
        //     ROS_INFO("Help needed: %s", help4 ? "Yes" : "No");
        // }   


    
        if (resource_client_.call(srv))
        {
            if (srv.response.success)
            {

                // Handle successful response
                Memory[0] = Memory[0] + srv.response.response.resource0;
                Memory[1] = Memory[1] + srv.response.response.resource1;
                Memory[2] = Memory[2] + srv.response.response.resource2;
                Memory[3] = Memory[3] + srv.response.response.resource3;
                Memory[4] = Memory[4] + srv.response.response.resource4;
                help0 = srv.response.help0;
                help1 = srv.response.help1;
                help2 = srv.response.help2;
                help3 = srv.response.help3;
                help4 = srv.response.help4;
                
                // ROS_INFO("Memory %.5f %.5f %.5f %.5f %.5f", Memory[0], Memory[1], Memory[2], Memory[3], Memory[4]);
                // ROS_INFO("Leader contribution %.10f %.10f %.10f %.10f %.10f", srv.response.response.resource0, srv.response.response.resource1, srv.response.response.resource2, srv.response.response.resource3, srv.response.response.resource4);
                // ROS_INFO("Help needed: %s", help ? "Yes" : "No");
            }
            else
            {
                ROS_WARN("Resource request failed");
                // Handle failure response
            }
        }
        else
        {
            ROS_ERROR("Failed to call service resource_service");
        }
    }
    
    double clamp(double value, double min_val, double max_val)
    {
        return std::max(min_val, std::min(value, max_val));
    }

    double computeBatteryCurrent(double linear_vel, double angular_vel, double dt)
    {
        double acceleration = 0.0;

        if (dt > 1e-3)
        {
            acceleration = (linear_vel - prev_linear_velocity_) / dt;
        }

        double rolling_force = rolling_coeff_ * robot_mass_ * 9.81;
        double total_force = robot_mass_ * acceleration + rolling_force;

        // Aggiunta di costo energetico dovuto alla rotazione
        double turning_force = std::abs(angular_vel) * 0.3;
        total_force += turning_force;

        if (total_force < 0.0)
        {
            total_force = 0.0;
        }

        double total_torque = total_force * wheel_radius_;
        double wheel_torque = total_torque / (4.0 * drivetrain_efficiency_);

        double wheel_current = wheel_torque / motor_kt_;

        double battery_current = 4.0 * wheel_current + aux_current_;

        // Penalizzazione ad alta velocità
        battery_current += 1.5 * std::abs(linear_vel);

        prev_linear_velocity_ = linear_vel;

        return std::max(0.0, battery_current);
    }

    void updateSimulatedBattery(double linear_vel, double angular_vel)
    {
        double current_time = ros::Time::now().toSec();

        if (prev_update_time_ == 0.0)
        {
            prev_update_time_ = current_time;
            return;
        }

        double dt = current_time - prev_update_time_;
        prev_update_time_ = current_time;

        battery_current_ = computeBatteryCurrent(linear_vel, angular_vel, dt);

        // Correzione capacità effettiva ad alte correnti
        double eta_I = 1.0;

        if (battery_current_ >= 1.0 && battery_current_ < 3.0)
        {
            eta_I = 0.97;
        }
        else if (battery_current_ >= 3.0)
        {
            eta_I = 0.93;
        }

        double effective_capacity = battery_capacity_ah_ * eta_I;

        // Coulomb counting
        battery_soc_ -= (battery_current_ * dt) / (3600.0 * effective_capacity);
        battery_soc_ = clamp(battery_soc_, 0.0, 1.0);

        // Modello Thevenin RC
        double alpha = std::exp(-dt / (R1_ * C1_));
        V_rc_ = alpha * V_rc_ + R1_ * (1.0 - alpha) * battery_current_;

        // OCV semplificata per 3 celle Li-Ion in serie
        double voc = 9.0 + 3.6 * battery_soc_;

        battery_voltage_ = voc - battery_current_ * R0_ - V_rc_;
        battery_voltage_ = clamp(battery_voltage_, 9.0, 12.6);

        battery_percentage_ = battery_soc_ * 100.0;

        sensor_msgs::BatteryState battery_msg;
        battery_msg.header.stamp = ros::Time::now();
        battery_msg.voltage = battery_voltage_;
        battery_msg.current = -battery_current_;
        battery_msg.percentage = battery_soc_;
        battery_msg.design_capacity = battery_capacity_ah_;
        battery_msg.present = true;

        battery_pub_.publish(battery_msg);
        //Priority[Priority.size() - 1] = -0.1 + (battery_voltage_ - volt_max)/(volt_min - volt_max)*1.1;
        Priority[Priority.size() - 1] = -0.1 + ((100.0f - battery_percentage_)/100.0f)*1.1;
        if (returning == true){
            Priority[Priority.size() - 1] = 1.5f; // Massima priorità per il ritorno alla stazione di ricarica
        }
    }

    
    // ========================
    // Loop principale
    // ========================
    void run()
    {
        int control_rate = 15;
        ros::Rate rate(control_rate);

        bool uptake = false;
        mission_start_time_ = ros::Time::now().toSec();
        ROS_INFO("Mission timer started: %.2f s", mission_start_time_);
        // {
        //     ROS_INFO("Init Pos %zu: x=%.2f y=%.2f z=%.2f yaw=%.2f",
        //              i,
        //              init_pos[i].x,
        //              init_pos[i].y,
        //              init_pos[i].z,
        //              init_pos[i].yaw);
        // }
        //sleep(10.0); // Ensure parameters are loaded before proceeding

        // Apply rotation to waypoints
        // double rotation_angle = M_PI; // Pi radians (180 degrees) around the y-axis
        // waypoints_ = rotateWaypoints(waypoints_, rotation_angle);
        
        while (ros::ok())
        {  
            if(mission_start_time_ == 0.0){
                mission_start_time_ = ros::Time::now().toSec();
                ROS_INFO("Mission timer started: %.2f s", mission_start_time_);
            }
            sensing_points = generateSurroundingPoints(radius);

            std::vector<double> perc_values = std::vector<double>(sensing_points.size());
            resulting_vectors = sensing_points;
            int j = 0;
            int i = 0;
            for (const auto& wp : waypoints_) {
                double distance = distance2D(robot[id].x, robot[id].y, wp.x, wp.y);

                double temp = gaussian(distance, wp_gvalue);
                if (distance < 1.0) // 2.5 sim
                {

                    uptake = true;
                    
                    discovered_[i] = true;
                }
                else
                {
                    uptake = false;
                }
                
                Priority[i] = biological_calculation(i, temp, uptake);
                //ROS_INFO("Priority[%d]: %.2f", i, Priority[i]);
                // ROS_INFO("Memory[%d]: %.5f", i, Memory[i]);
                i = i + 1;
            }
            //ROS_INFO("Priority[%d]: %.2f", i, Priority[i]);
            // Timer 1: si ferma quando tutte le priorità tranne l'ultima sono > 0.0
            if (!mission_timer_stopped_ &&
                std::all_of(Priority.begin(), Priority.end() - 1,
                            [](double p)
                            {
                                return p < 0.6;
                            }))
            {
                mission_completion_time_ = time;
                mission_timer_stopped_ = true;

                ROS_INFO("Mission completion timer stopped: %.2f s",
                        mission_completion_time_);
            }
            for (const auto& point : sensing_points)
            {
                if (Priority[Priority.size() - 1] < 0.6f && std::any_of(Priority.begin(), Priority.end() - 1, [](double p){ return p > 0.6f; }))
                {
                int k = 0;
                // ROS_INFO("Rosbot %i exploring", id);
                for (const auto& wp : waypoints_) {
                    //ROS_INFO("Waypoint: x=%.2f y=%.2f z=%.2f yaw=%.2f", wp.x, wp.y, wp.z, wp.yaw);
                    
                    if (discovered_[k] == false){
                        Priority[k] = 0.0; // Bonus per waypoint non ancora scoperto
                    }
                    double distance = distance2D(point.x, point.y, wp.x, wp.y);
                    perc_values[j] = perc_values[j] + gaussian(distance, wp_gvalue)*Priority[k];
                    //ROS_INFO("perc_value %.2f", perc_values[j]);
                    k = k + 1;
                }
                }
                else{
                    returning = true; 
                    if (!charging_timer_stopped_)
                    {
                        ROS_INFO("Rosbot %i returning home", id);
                        for (const auto& station : charging_stations)
                        {
                            double dist_station = distance2D(robot[id].x,
                                                            robot[id].y,
                                                            station.x,
                                                            station.y);
        
                            if (dist_station < 1.0) // soglia arrivo charging station
                            {
                                charging_completion_time_ = ros::Time::now().toSec() - mission_start_time_;
                                charging_timer_stopped_ = true;
        
                                ROS_INFO("Charging station timer stopped: %.2f s",
                                        charging_completion_time_);
                                break;
                            }
                        }
                    }
                }
                //ROS_INFO("1valori percepiti in ciascun punto %.5f (%.2f,%.2f)", perc_values[j], point.x, point.y);
                //ROS_INFO("Obstacle size %i", obs_positions.size());
                if (obs_positions.empty())
                {
                    //ROS_WARN("No obstacles detected, skipping obstacle processing.");
                    ;
                }
                else
                {
                    for (const auto& obstacle : obs_positions) {
                        //ROS_INFO("Obstacle position: x=%.2f y=%.2f, vector lenght %i", obstacle.x, obstacle.y, obs_positions.size());
                        double distance = distance2D(point.x, point.y, obstacle.x, obstacle.y);
                        perc_values[j] = perc_values[j] - gaussian(distance, obs_gvalue)/obs_positions.size();
                        
                    }
                    // ROS_INFO("2valori percepiti in ciascun punto %.5f (%.2f,%.2f)", perc_values[j], point.x, point.y);
                }
                int o = 0;
                for (const auto& robot_ : robot){
                    if (o == id){
                        // ROS_INFO("Current robot position (run): x=%.2f, y=%.2f, yaw=%.2f", robot_.x, robot_.y, robot_.yaw) ;
                        ;
                    }
                    else{
                    
                    double distance = distance2D(point.x, point.y, robot_.x, robot_.y);
                    perc_values[j] = perc_values[j] - gaussian(distance, drones_gvalue);
                    }
                    o = o + 1;
                }
                for (const auto& station : charging_stations) {
                    double distance = distance2D(point.x, point.y, station.x, station.y);
                    perc_values[j] = perc_values[j] + gaussian(distance, docks_gvalue)*Priority[Priority.size()-1];
                }
                resulting_vectors[j].x = (point.x - robot[id].x) * perc_values[j];
                resulting_vectors[j].y = (point.y - robot[id].y) * perc_values[j];
                //ROS_INFO("3valori percepiti in ciascun punto %.5f (%.2f,%.2f)", perc_values[j], point.x, point.y);
                //ROS_INFO("resulting vectors: %.5f, %.5f", resulting_vectors[j].x, resulting_vectors[j].y );
                j = j + 1;
            }
            Point2D sum;
            sum.x = 0.0;
            sum.y = 0.0;
            sendRequest();
            for (const auto& vec : resulting_vectors)
            {
                sum.x = sum.x + vec.x;
                sum.y = sum.y + vec.y;
            }
            sum.x = sum.x * control_rate;
            sum.y = sum.y * control_rate;

            //ROS_INFO("Resulting vector sum: x=%.20f, y=%.20f", sum.x, sum.y);
            double norm = std::sqrt(sum.x * sum.x + sum.y * sum.y);
            if (std::isnan(norm)) {
                ROS_WARN("Norm is NaN, setting to 0.0");
                norm = 0.0;
            }
            // ROS_INFO("Norm of resulting vector sum: %.20f", norm);
            
            double yaw = std::atan2(sum.y, sum.x);
            if (std::isnan(yaw)) {
                ROS_WARN("Yaw is NaN, setting to 0.0");
                yaw = 0.0;
            }
            // ROS_INFO("Yaw voluto %.2f", yaw);
            // ROS_INFO("Number of obstacles detected: %lu", obs_positions.size());
            geometry_msgs::Twist vel_msg;
            if (yaw > M_PI){
                yaw = yaw - 2*M_PI;
            }
            if (yaw <= -M_PI){
                yaw = yaw + 2*M_PI;
            }
            if (robot[id].yaw > M_PI){
                robot[id].yaw = robot[id].yaw - 2*M_PI;
            }       
            if (robot[id].yaw <= -M_PI){
                robot[id].yaw = robot[id].yaw + 2*M_PI;
            }
            double error = yaw - robot[id].yaw;
            if (error > M_PI){
                error = error - 2*M_PI;
            }
            if (error <= -M_PI){
                error = error + 2*M_PI;
            }
            double scale  = 1.0;
            if (norm != 0.0 && norm > max_vel){
                scale = max_vel / norm;
            }
            //ROS_INFO("Absolute error: %.2f %.2f", error, M_PI/6);
            if (error < M_PI/8 && error > -M_PI/8) {
                vel_msg.angular.z = pidYawControl(error, 0.5, 0.01, 0.05);
                vel_msg.linear.x = max_vel;
                //ROS_INFO("DRITTO");
            }
            else {
                vel_msg.linear.x = norm * scale;
                vel_msg.angular.z = pidYawControl(error, 0.5, 0.01, 0.05);
                //ROS_INFO("CURVO");
            }
            // vel_msg.angular.z = pidYawControl(error, 1, 0.0, 0.0);
            // vel_msg.linear.x = norm * scale ;
            velocity_pub_.publish(vel_msg);
            if (simulation_mode_)
            {
                updateSimulatedBattery(vel_msg.linear.x, vel_msg.angular.z);
            }
            for (int i = 0; i < Memory.size(); i++)
            {
                Memory[i] = Memory[i] + Memory_temp[i];
                if (Memory[i] > 0.0005253){
                    Memory[i] = 0.0005253;
                }
            }
            
            geometry_msgs::Pose goal;
            goal.position.x = robot[id].x;
            goal.position.y = robot[id].y;
            goal.position.z = 0.0;
            goal.orientation.w = std::cos(yaw * 0.5);
            goal.orientation.z = std::sin(yaw * 0.5);
            pose_pub.publish(goal);
            // Timer 2: si ferma quando il robot arriva a una charging station

            time = ros::Time::now().toSec() - mission_start_time_;

            // POSIZIONE
            pos_file_ << std::fixed << std::setprecision(5)
                    << time << ","
                    << robot[id].x << ","
                    << robot[id].y << ","
                    << robot[id].yaw << "\n";

            // PRIORITÀ
            priority_file_ << time;
            for (const auto& p : Priority)
                priority_file_ << "," << p;
            priority_file_ << "\n";

            // DIREZIONE (vettore risultante)
            direction_file_ << time << ","
                            << sum.x << ","
                            << sum.y << ","
                            << yaw << "\n";
            battery_file_ << std::fixed << std::setprecision(5)
                          << time << ","
                          << battery_voltage_ << ","
                          << battery_percentage_ << "\n";
            if (!mission_time_saved_ &&
                mission_timer_stopped_ &&
                charging_timer_stopped_)
            {
                mission_time_file_ << mission_completion_time_
                                    << ","
                                    << charging_completion_time_
                                    << "\n";
            
                mission_time_saved_ = true;
            }
            //obs_positions.clear();
            ros::spinOnce();
            
            rate.sleep();
            pos_file_.flush();
            priority_file_.flush();
            direction_file_.flush();
            battery_file_.flush();

        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "indoor_bot_node");

    std::string waypoints_file =
        "/root/catkin_ws/src/indoor_bot/config/waypoints.yaml";

    IndoorBot bot(waypoints_file);
    bot.run();

    return 0;
}