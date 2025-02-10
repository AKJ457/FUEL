#include <fstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/frontier_finder.h>
#include <plan_manage/planner_manager.h>
#include <exploration_manager/expl_data.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

using namespace Eigen;

namespace fast_planner {

// Constructor
FastExplorationManager::FastExplorationManager() {}

// Destructor
FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

// Initialize function
void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);
}

// Main exploration motion planning function
int FastExplorationManager::planExploreMotion(const Vector3d& pos, const Vector3d& vel, 
                                              const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  ed_->views_.clear();
  ed_->global_tour_.clear();

  // Search frontiers
  frontier_finder_->searchFrontiers();

  // Get viewpoints
  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);

  if (ed_->frontiers_.empty()) {
    ROS_WARN("No coverable frontier.");
    return NO_FRONTIER;
  }

  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
  for (size_t i = 0; i < ed_->points_.size(); ++i)
    ed_->views_.push_back(ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

  Vector3d next_pos;
  double next_yaw;
  if (ed_->points_.size() > 1) {
    vector<int> indices;
    findGlobalTour(pos, vel, yaw, indices);

    if (ep_->refine_local_) {
      vector<vector<Vector3d>> dummy_n_points;
      vector<vector<double>> dummy_n_yaws;
      vector<Vector3d> refined_pts;
      vector<double> refined_yaws;

      refineLocalTour(pos, vel, yaw, dummy_n_points, dummy_n_yaws, refined_pts, refined_yaws);
      next_pos = refined_pts[0];
      next_yaw = refined_yaws[0];

    } else {
      next_pos = ed_->points_[indices[0]];
      next_yaw = ed_->yaws_[indices[0]];
    }
  } else if (ed_->points_.size() == 1) {
    next_pos = ed_->points_[0];
    next_yaw = ed_->yaws_[0];
  } else {
    ROS_ERROR("Empty destination.");
  }

  planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_);
  return SUCCEED;
}

// Global tour computation
void FastExplorationManager::findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, 
                                            const Vector3d cur_yaw, vector<int>& indices) {
  auto t1 = ros::Time::now();
  
  vector<double> rewards(ed_->frontiers_.size());
  for (size_t i = 0; i < ed_->frontiers_.size(); i++) {
    rewards[i] = computeMDPReward(ed_->frontiers_[i], cur_pos, cur_vel, cur_yaw);
  }

  indices = solveMDP(rewards, ed_->frontiers_);
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);
}

// Compute MDP reward function
double FastExplorationManager::computeMDPReward(const vector<Vector3d>& frontier, 
                                                const Vector3d& cur_pos,
                                                const Vector3d& cur_vel, const Vector3d& cur_yaw) {
  double info_gain = computeInformationGain(frontier);
  double cost = (cur_pos - frontier[0]).norm();
  return info_gain - 0.5 * cost;
}

// Compute information gain function
double FastExplorationManager::computeInformationGain(const vector<Vector3d>& frontier) {
  double info_gain = 0.0;
  for (const auto& point : frontier) {
    double distance = edt_environment_->sdf_map_->getDistance(point);
    if (distance < 0) distance = 0;
    info_gain += 1.0 / (1.0 + distance);
  }
  return info_gain;
}

// Solve MDP for best exploration path
vector<int> FastExplorationManager::solveMDP(const vector<double>& rewards, 
                                             const vector<vector<Vector3d>>& frontiers) {
  vector<int> best_indices;
  double max_reward = -1e9;
  int best_idx = 0;
  for (size_t i = 0; i < rewards.size(); i++) {
    if (rewards[i] > max_reward) {
      max_reward = rewards[i];
      best_idx = i;
    }
  }
  best_indices.push_back(best_idx);
  return best_indices;
}

// Path shortening function
void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) return;
  const double dist_thresh = 3.0;
  vector<Vector3d> short_tour = { path.front() };

  for (size_t i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) 
    short_tour.push_back(path.back());

  path = short_tour;
}

// Generate trajectory function
bool FastExplorationManager::generateTrajectory(const Vector3d& start_pos, const Vector3d& start_vel, 
                                                const Vector3d& start_acc, const Vector3d& goal_pos, 
                                                const Vector3d& goal_vel) {
  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(start_pos, goal_pos) != Astar::REACH_END) {
    ROS_ERROR("No valid path found!");
    return false;
  }
  vector<Vector3d> path = planner_manager_->path_finder_->getPath();
  shortenPath(path);
  planner_manager_->planExploreTraj(path, start_vel, start_acc, 1.0);
  return true;
}
} // namespace fast_planner
