#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace fast_planner {
class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
class FrontierFinder;
struct ExplorationParam;
struct ExplorationData;

enum EXPL_RESULT { NO_FRONTIER, FAIL, SUCCEED };

class FastExplorationManager {
public:
  FastExplorationManager();
  ~FastExplorationManager();

  void initialize(ros::NodeHandle& nh);

  // Main function to execute exploration planning
  int planExploreMotion(const Vector3d& pos, const Vector3d& vel, 
                        const Vector3d& acc, const Vector3d& yaw);

  // Classic and rapid frontier methods
  int classicFrontier(const Vector3d& pos, const double& yaw);
  int rapidFrontier(const Vector3d& pos, const Vector3d& vel, 
                    const double& yaw, bool& classic);

  // Exploration execution function (set as public)
  bool executeExploration(const Vector3d& pos, const Vector3d& vel, 
                          const Vector3d& acc, const Vector3d& yaw);

  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FrontierFinder> frontier_finder_;

private:
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;

  // Compute information gain at a given frontier
  double computeInformationGain(const vector<Vector3d>& frontier);

  // Compute reward function for MDP-based exploration
  double computeMDPReward(const vector<Vector3d>& frontier, 
                          const Vector3d& cur_pos,
                          const Vector3d& cur_vel, 
                          const Vector3d& cur_yaw);

  // Solve MDP to determine the best exploration sequence
  vector<int> solveMDP(const vector<double>& rewards, 
                       const vector<vector<Vector3d>>& frontiers);

  // Generate an optimal tour based on frontiers
  void findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, 
                      const Vector3d cur_yaw, vector<int>& indices);

  // Refine local tour for better exploration efficiency
  void refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, 
                       const Vector3d& cur_yaw, 
                       const vector<vector<Vector3d>>& n_points, 
                       const vector<vector<double>>& n_yaws,
                       vector<Vector3d>& refined_pts, 
                       vector<double>& refined_yaws);

  // Shorten exploration paths to reduce redundant waypoints
  void shortenPath(vector<Vector3d>& path);

  // Generate a smooth trajectory for UAV exploration
  bool generateTrajectory(const Vector3d& start_pos, const Vector3d& start_vel, 
                          const Vector3d& start_acc, const Vector3d& goal_pos, 
                          const Vector3d& goal_vel);

public:
  typedef shared_ptr<FastExplorationManager> Ptr;
};

}  // namespace fast_planner

#endif  // _EXPLORATION_MANAGER_H_
