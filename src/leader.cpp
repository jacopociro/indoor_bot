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
        }
        start_time = ros::Time::now();

        std::string base_path = "/root/catkin_ws/src/indoor_bot/indoor_bot/data/";

        memory_file.open(base_path + "memory_log.csv");
        ph_file.open(base_path + "photosynthesis_log.csv");

        // header CSV
        memory_file << "time";
        for (size_t i = 0; i < Memory.size(); i++)
        {
            memory_file << ",m" << i;
        }
        memory_file << std::endl;

        ph_file << "time,ph" << std::endl;
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
            if (Memory[i] != 0.0f)
                check[i] = true;

            if (check[i] && Memory[i] < min_mem)
                min_mem = Memory[i];
        }

        if (min_mem == 100.0f)
            min_mem = 0.0f; // If no memory is used, set to 0

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
        ph_file << std::fixed << std::setprecision(5)
                << t << "," << ph << std::endl;
    }
    void run()
    {
        while(ros::ok())
        {

            ros::spinOnce();
        }

    }

    bool handleService(const std::string& ugv_name,
                    indoor_bot::resourcesrv::Request &req,
                    indoor_bot::resourcesrv::Response &res)
    {
        
        Memory[0] = Memory[0] + req.request.resource0;
        Memory[1] = Memory[1] + req.request.resource1;
        Memory[2] = Memory[2] + req.request.resource2;
        Memory[3] = Memory[3] + req.request.resource3;
        Memory[4] = Memory[4] + req.request.resource4;
        //ROS_INFO("Received resource request by %s: laeder memory:%f, %f, %f, %f, %f",
            // ugv_name.c_str(),
            // Memory[0],
            // Memory[1],
            // Memory[2],
            // Memory[3],
            // Memory[4]);
        res.success = true;
        if (Memory[0] == 0.0) {
            res.help0 = true;
        }
        else{
            res.help0 = false;
        }
        if (Memory[1] == 0.0) {
            res.help1 = true;
        }
        else{
            res.help1 = false;
        }
        if (Memory[2] == 0.0) {
            res.help2 = true;
        }
        else{
            res.help2 = false;
        }
        if (Memory[3] == 0.0) {
            res.help3 = true;
        }
        else{
            res.help3 = false;
        }
        if (Memory[4] == 0.0) {
            res.help4 = true;
        }
        else{
            res.help4 = false;
        }

        float ph = photosynthesis_prod();
        a = a + ph;
        saveToCSV(a);
        res.response.resource0 = req.request.resource0*split/n_UGV;
        res.response.resource1 = req.request.resource1*split/n_UGV;
        res.response.resource2 = req.request.resource2*split/n_UGV;
        res.response.resource3 = req.request.resource3*split/n_UGV;
        res.response.resource4 = req.request.resource4*split/n_UGV;

        Memory[0] = Memory[0] - req.request.resource0*split/n_UGV;
        Memory[1] = Memory[1] - req.request.resource1*split/n_UGV;
        Memory[2] = Memory[2] - req.request.resource2*split/n_UGV;
        Memory[3] = Memory[3] - req.request.resource3*split/n_UGV;
        Memory[4] = Memory[4] - req.request.resource4*split/n_UGV;


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