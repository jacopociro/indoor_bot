#include "ros/ros.h"
#include "indoor_bot/resourcesrv.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iomanip>

using namespace boost::placeholders;
struct Pose {
    double x, y, z, yaw;
};

class ResourceServer
{
private:
    ros::NodeHandle nh_;
    std::vector<ros::ServiceServer> service_list;

    std::vector<Pose> waypoints;
    float split = 0.95;
    bool help = false;
    std::vector<double> Memory;
    std::vector<bool> check;
    int n_UGV;
    std::vector<std::string> ugv_names;
    float a = 0.0f;
    std::ofstream memory_file;
    std::ofstream ph_file;
    ros::Time start_time;
    std::ofstream mission_time_file;
    bool mission_completed = false;
    double mission_start_time_ = 0.0;
    float send_back = 1.0f;
    std::vector<double> quota;
    std::vector<double> value;
    
public:
    ResourceServer(const std::string& waypoint_file)
     : nh_()
    {
        // PARAMETERS
        nh_.param("n_UGV", n_UGV, 1);
        nh_.getParam("/UGV_names", ugv_names);
        waypoints = readWaypoints(waypoint_file);
        for (const auto& wp : waypoints)
        {
            Memory.push_back(0.0);
            check.push_back(false);
        }
        for (const auto& name : ugv_names)
        {
            boost::function<bool(indoor_bot::resourcesrv::Request&, indoor_bot::resourcesrv::Response&)> service_callback
                = boost::bind(&ResourceServer::handleService, this, name, _1, _2);
            ros::ServiceServer service = nh_.advertiseService(name + "/resource_service",
                service_callback);
            service_list.push_back(service);
            quota.push_back(1.0f /n_UGV);
            value.push_back(0.0f);
        }
        //start_time = ros::Time::now();

        std::string base_path = "/root/catkin_ws/src/indoor_bot/indoor_bot/data/";

        memory_file.open(base_path + "memory_log.csv");
        ph_file.open(base_path + "photosynthesis_log.csv");
        mission_time_file.open(base_path + "mission_time.csv");
        mission_time_file << "mission_time"<< std::endl;
        // header CSV
        memory_file << "time";
        for (size_t i = 0; i < Memory.size(); i++)
        {
            memory_file << ",m" << i;
        }
        memory_file << std::endl;

        ph_file << "time,ph" << std::endl;
    }
    void checkMissionCompletion()
    {
        if (mission_completed)
            return;

        bool all_positive = true;

        for (const auto& m : Memory)
        {
            if (m <= 0.0)
            {
                all_positive = false;
                break;
            }
        }

        if (all_positive || mission_completed == false)
        {
            double mission_time = (ros::Time::now() - start_time).toSec();

            mission_time_file << std::fixed << std::setprecision(5)
                              << mission_time << std::endl;

            ROS_INFO("Mission completed in %.5f seconds", mission_time);

            mission_completed = true;
        }
    }
    // --- Funzione photosynthesis_prod ---
    float photosynthesis_prod()
    {
        float lambda = 0.82f;
        float a_max = 6 * (24 - 18);   // 18 hours of light
        float ph_max = 12.7f;
        float C = lambda + (1 - lambda) * (a_max - a) / a_max;

        float min_mem = 100.0f; // Initialize to a large value

        for (size_t i = 0; i < Memory.size(); i++)
        {
            if (Memory[i] > 0.0f){
                    check[i] = true;
                }
            else{
                check[i] = false;
            }
            if (Memory[i] < min_mem && Memory[i] > 0.0f){
                min_mem = Memory[i];
            }
        }

        if (min_mem == 100.0f)
            {min_mem = 0.0f;} // If no memory is used, set to 0

        for (size_t i = 0; i < Memory.size(); i++)
        {
            Memory[i] -= min_mem;
        }

        float ph_prod = ph_max * min_mem * C;
        return ph_prod;
    }
    void saveToCSV(float ph)
    {
        double t = (ros::Time::now() - start_time).toSec();

        // salva Memory
        memory_file << std::fixed << std::setprecision(8) << t;
        for (const auto& m : Memory)
        {
            memory_file << "," << m;
        }
        memory_file << std::endl;

        // salva photosynthesis
        ph_file << std::fixed << std::setprecision(10)
                << t << "," << ph << std::endl;
    }
    void run()
    {
        start_time = ros::Time::now();
        quota = {1.0f/2.0f, 1.0f/3.0f, 1.0f/6.0f};
        //quota = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
        while(ros::ok())
        {
            send_back = 0.0000001f;
            
            // value = {0.0f, 0.0f, 0.0f};
            mission_start_time_ = ros::Time::now().toSec();
            ros::spinOnce();
            
        }

    }

    bool handleService(const std::string& ugv_name,
                    indoor_bot::resourcesrv::Request &req,
                    indoor_bot::resourcesrv::Response &res)
    {
        //ROS_INFO("send back: %.10f", send_back);
        res.success = true;
        
        if (req.request.resource0 < send_back && req.request.resource0 > 0.0f){
            send_back = req.request.resource0;
        }
        if (req.request.resource1 < send_back && req.request.resource1 > 0.0f){
            send_back = req.request.resource1;
        }
        if (req.request.resource2 < send_back && req.request.resource2 > 0.0f){
            send_back = req.request.resource2;
        }
        if (req.request.resource3 < send_back && req.request.resource3 > 0.0f){
            send_back = req.request.resource3;
        }
        if (req.request.resource4 < send_back && req.request.resource4 > 0.0f){
            send_back = req.request.resource4;
        }
        if (ugv_name == "rosbot_1"){
            value[0] = send_back;
            int index = 0;
        }
        else if (ugv_name == "rosbot_2"){
            value[1] = send_back;
            int index = 1;
        }
        else if (ugv_name == "rosbot_3"){
            value[2] = send_back;
            int index = 2;
        }
        auto valore = std::max_element(value.begin(), value.end());
        auto valore_min = std::min_element(value.begin(), value.end());
        int it = valore - value.begin();
        if (ugv_name == "rosbot_1"){
            
            if (*valore == value[0]){
                it = 0;
            }
            else if (*valore_min == value[0]){
                it = 2;
            }
            else {
                it = 1;
            }
        }
        else if (ugv_name == "rosbot_2"){
            
            if (*valore == value[1]){
                it = 0;
            }
            else if (*valore_min == value[1]){
                it = 2;
            }
            else {
                it = 1;
            }
        }
        else if (ugv_name == "rosbot_3"){
            
            if (*valore == value[2]){
                it = 0;
            }
            else if (*valore_min == value[2]){
                it = 2;
            }
            else {
                it = 1;
            }
        }
        if (req.request.resource0 < 1000.5){
            Memory[0] = Memory[0] + req.request.resource0;
            if (Memory[0] <= 0.0f) {
                res.response.resource0 = 0.0f;
            }
            else{
                res.response.resource0 = *valore*split*quota[it];
                Memory[0] = Memory[0] - *valore*split*quota[it];
            }
            

        }
        if (req.request.resource1 < 1000.5){
            Memory[1] = Memory[1] + req.request.resource1;
            if (Memory[1] <= 0.00000012f) {
                res.response.resource1 = 0.0f;
            }
            else{
                res.response.resource1 = *valore*split*quota[it];
                Memory[1] = Memory[1] - *valore*split*quota[it];
            }
            
        }
        if (req.request.resource2 < 1000.5){
            Memory[2] = Memory[2] + req.request.resource2;
            if (Memory[2] <= 0.00000012f) {
                res.response.resource2 = 0.0f;
            }
            else{
                res.response.resource2 = *valore*split*quota[it];
                Memory[2] = Memory[2] - *valore*split*quota[it];
            }
            
        }
        if (req.request.resource3 < 1000.5){
            Memory[3] = Memory[3] + req.request.resource3;
            if (Memory[3] <= 0.00000012f) {
                res.response.resource3 = 0.0f;
            }
            else{
                res.response.resource3 = *valore*split*quota[it];
                Memory[3] = Memory[3] - *valore*split*quota[it];
            }
            
        }
        if (req.request.resource4 < 1000.5){
            Memory[4] = Memory[4] + req.request.resource4;
            if (Memory[4] <= 0.00000012f) {
                res.response.resource4 = 0.0f;
            }
            else{
                res.response.resource4 = *valore*split*quota[it];
                Memory[4] = Memory[4] - *valore*split*quota[it];
            }
            
        }

        // ROS_INFO("Received resource request by %s: laeder memory:%f, %f, %f, %f, %f",
        //     ugv_name.c_str(),
        //     Memory[0],
        //     Memory[1],
        //     Memory[2],
        //     Memory[3],
        //     Memory[4]);
        res.success = true;
        if (Memory[0] <= 0.0) {
            res.help0 = true;
        }
        else{
            res.help0 = false;
        }
        if (Memory[1] <= 0.0) {
            res.help1 = true;
        }
        else{
            res.help1 = false;
        }
        if (Memory[2] <= 0.0) {
            res.help2 = true;
        }
        else{
            res.help2 = false;
        }
        if (Memory[3] <= 0.0) {
            res.help3 = true;
        }
        else{
            res.help3 = false;
        }
        if (Memory[4] <= 0.0) {
            res.help4 = true;
        }
        else{
            res.help4 = false;
        }
        float ph = 0.0f;
        if (std::all_of(Memory.begin(), Memory.end(), [](double m) { return m > 0.0; })) {
            ph = photosynthesis_prod();
        }

        a = a + ph;
        saveToCSV(a);
        
        
        
        
        
        ROS_INFO("Leader contribution to %s: %.10f, %.10f, %.10f, %.10f, %.10f [LEADER SIDE] \n max value: %.10f, quota: %.10f",
            ugv_name.c_str(),
            res.response.resource0,
            res.response.resource1,
            res.response.resource2,
            res.response.resource3,
            res.response.resource4,
            *valore,
            quota[it]);
        
        ROS_INFO("Values of 'value': %.10f, %.10f, %.10f",
            value[0], value[1], value[2]);
        
        
        
        checkMissionCompletion();

        return true;
    }
    std::vector<Pose> readWaypoints(const std::string& filename)
    {
        std::vector<Pose> waypoints;

        try {
            YAML::Node config = YAML::LoadFile(filename);

            if (config["wp"]) {
                for (const auto& wp : config["wp"]) {
                    Pose waypoint;
                    waypoint.x   = wp["x"].as<double>();
                    waypoint.y   = wp["y"].as<double>();
                    waypoint.z   = wp["z"].as<double>();
                    waypoint.yaw = wp["yaw"].as<double>();
                    waypoints.push_back(waypoint);
                }
            }
        }
        catch (const YAML::Exception& e) {
            ROS_ERROR("Errore lettura YAML: %s", e.what());
        }

        return waypoints;
    }
    ~ResourceServer()
    {
        if (memory_file.is_open())
            memory_file.close();
    
        if (ph_file.is_open())
            ph_file.close();
        if (mission_time_file.is_open())
            mission_time_file.close();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "resource_server");
    std::string waypoints_file =
        "/root/catkin_ws/src/indoor_bot/config/waypoints.yaml";

    ResourceServer server(waypoints_file);

    server.run();
    return 0;
}