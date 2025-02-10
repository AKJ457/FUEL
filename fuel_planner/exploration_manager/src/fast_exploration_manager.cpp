// #include <fstream>
#include <fstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
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

// SECTION interfaces for setup and query
FastExplorationManager::FastExplorationManager() {}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

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

int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
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
  for (int i = 0; i < ed_->points_.size(); ++i)
    ed_->views_.push_back(
        ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

  Vector3d next_pos;
  double next_yaw;
  if (ed_->points_.size() > 1) {
    vector<int> indices;
    findGlobalTour(pos, vel, yaw, indices);

    if (ep_->refine_local_) {
      refineLocalTour(pos, vel, yaw, indices, next_pos, next_yaw);
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

// ✅ **MDP 기반 글로벌 탐색 경로 생성 함수**
void FastExplorationManager::findGlobalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    vector<int>& indices) {
  
  auto t1 = ros::Time::now();
  
  vector<double> rewards(ed_->frontiers_.size());
  for (size_t i = 0; i < ed_->frontiers_.size(); i++) {
    rewards[i] = computeMDPReward(ed_->frontiers_[i], cur_pos, cur_vel, cur_yaw);
  }

  indices = solveMDP(rewards, ed_->frontiers_);
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);
}

// ✅ **탐색 정보 이득을 고려한 보상 함수**
double FastExplorationManager::computeMDPReward(
    const vector<Vector3d>& frontier, const Vector3d& cur_pos,
    const Vector3d& cur_vel, const Vector3d& cur_yaw) {
  
  double info_gain = computeInformationGain(frontier);
  double cost = (cur_pos - frontier[0]).norm();
  return info_gain - 0.5 * cost;
}

// ✅ **🚀 추가해야 할 `computeInformationGain()` 함수**
double FastExplorationManager::computeInformationGain(const vector<Vector3d>& frontier) {
  double info_gain = 0.0;

  for (const auto& point : frontier) {
    double distance = edt_environment_->sdf_map_->getDistance(point);
    if (distance < 0) distance = 0;  // 음수 방지

    // 거리가 멀면 정보 이득이 낮음
    info_gain += 1.0 / (1.0 + distance);
  }

  return info_gain;
}
// ✅ **MDP 기반 최적 경로 결정 함수**
vector<int> FastExplorationManager::solveMDP(
    const vector<double>& rewards, const vector<vector<Vector3d>>& frontiers) {
  
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

// ✅ **지역 탐색 경로 개선 함수**
void FastExplorationManager::refineLocalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<int>& indices, Vector3d& next_pos, double& next_yaw) {

  vector<Vector3d> candidate_positions;
  vector<double> candidate_yaws;
  
  for (int i = 0; i < min((int)indices.size(), ep_->refined_num_); i++) {
    candidate_positions.push_back(ed_->points_[indices[i]]);
    candidate_yaws.push_back(ed_->yaws_[indices[i]]);
  }

  int best_idx = 0;
  double min_cost = 1e9;
  for (int i = 0; i < candidate_positions.size(); i++) {
    double cost = (cur_pos - candidate_positions[i]).norm();
    if (cost < min_cost) {
      min_cost = cost;
      best_idx = i;
    }
  }

  next_pos = candidate_positions[best_idx];
  next_yaw = candidate_yaws[best_idx];
}
// ✅ 탐색 경로 단순화 함수
void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  const double dist_thresh = 3.0;
  vector<Vector3d> short_tour = { path.front() };

  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // 장애물 회피를 위해 Waypoint 추가
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  if ((path.back() - short_tour.back()).norm() > 1e-3) 
    short_tour.push_back(path.back());

  // 경로 최소 3개 점 보장
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
}
// ✅ 탐색 경로를 UAV가 이동할 수 있도록 궤적 생성
bool FastExplorationManager::generateTrajectory(
    const Vector3d& start_pos, const Vector3d& start_vel, const Vector3d& start_acc,
    const Vector3d& goal_pos, const Vector3d& goal_vel) {
  
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
// ✅ 탐색 전략 개선
bool FastExplorationManager::executeExploration(const Vector3d& pos, const Vector3d& vel, 
                                                const Vector3d& acc, const Vector3d& yaw) {
  vector<int> global_indices;
  findGlobalTour(pos, vel, yaw, global_indices);

  if (global_indices.empty()) {
    ROS_WARN("No valid global tour found!");
    return false;
  }

  Vector3d next_pos;
  double next_yaw;
  refineLocalTour(pos, vel, yaw, global_indices, next_pos, next_yaw);

  return generateTrajectory(pos, vel, acc, next_pos, Vector3d(0, 0, 0));
}
int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);
  bool success = expl_manager_->executeExploration(fd_->start_pt_, fd_->start_vel_, 
                                                   fd_->start_acc_, fd_->start_yaw_);

  if (!success) {
    ROS_WARN("Exploration planning failed!");
    return FAIL;
  }

  planner_manager_->local_data_.start_time_ = ros::Time::now();
  return SUCCEED;
}
