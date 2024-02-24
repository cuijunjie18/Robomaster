#include <enemy_predictor/enemy_predictor.h>

using namespace enemy_predictor;

void Enemy::add_armor(TargetArmor &armor) {
    RCLCPP_INFO(predictor->get_logger(), "[add_armor] add_armor: %d", armor_cnt);
    static std::vector<int> alive_indexs(armor_cnt);
    static std::vector<int> absent_indexs(armor_cnt);
    alive_indexs.clear();
    absent_indexs.clear();
    for (int i = 0; i < (int)armors.size(); ++i) {
        if (armors[i].status == Status::Alive) {
            alive_indexs.push_back(i);
        }
    }
    for (int i = 0; i < (int)armors.size(); ++i) {
        if (armors[i].status == Status::Absent) {
            absent_indexs.push_back(i);
        }
    }

    // 在所有装甲板中寻找tracking armor并更新phase
    bool has_tracking_in_enemy = false;
    for (int i = 0; i < (int)armors.size(); ++i) {
        // 之前存在tracking
        if (armors[i].status == Status::Alive) {
            if (check_left(armor.getpos_pyd(), armors[i].getpos_pyd())) {
                armor.phase_in_enemy = (armors[i].phase_in_enemy - 1 + armor_cnt) % armor_cnt;
            } else {
                armor.phase_in_enemy = (armors[i].phase_in_enemy + 1 + armor_cnt) % armor_cnt;
            }
            has_tracking_in_enemy = true;
            break;
        }
    }
    if (!has_tracking_in_enemy) {
        armor.phase_in_enemy = 0;
    }

    // 没有活动装甲板
    if (alive_indexs.size() == 0) {
        armors.clear();

        // 跑出相机视野，已有装甲板清空，不清楚先前主副状态的失踪状态
        sub_tracking_absent_flag = true;
        tracking_absent_flag = true;

        armor_appear(armor);
        armors.push_back(armor);
        // new_armor_phase_id = armor.phase_id;
    } else if (alive_indexs.size() == 1) {
        // 有一个原有的装甲板
        //  sizeof(TargetArmor)
        TargetArmor previous_armor = armors[alive_indexs[0]];
        // 求解两个装甲板之间的位置坐标距离
        double armor_dis = calc_surface_dis_xyz(previous_armor.getpos_xyz(), armor.getpos_xyz());
        // add_armor_logger.sinfo("armor_dis: {}", armor_dis);
        // 两个装甲板之间的距离要小于阈值
        if (armor_dis < predictor->params.robot_2armor_dis_thresh) {
            // 成功组成一对装甲板
            armors.clear();
            // 1->2 switch
            // 保证在左边装甲板位于数组的前位
            if (check_left(previous_armor.getpos_pyd(), armor.getpos_pyd())) {
                armors.push_back(previous_armor);
                armor_appear(armor);
                armors.push_back(armor);
            } else {
                armor_appear(armor);
                armors.push_back(armor);
                armors.push_back(previous_armor);
            }
        } else {
            if (previous_armor.getpos_xyz().norm() > armor.getpos_xyz().norm()) {
                armors.clear();
                armors.push_back(armor);
                armor_appear(armor);
            }
        }
    } else if (alive_indexs.size() == 2) {
        // add_armor_logger.warn("3 armors");
        // TODO
        RCLCPP_INFO(predictor->get_logger(), "[add_armor] 3 armors!");
        return;
    } else {
        RCLCPP_INFO(predictor->get_logger(), "[add_armor] impossible armor amount: %d!", armor_cnt);
        // 异常情况
        armors.clear();
        armor_appear(armor);
        armors.push_back(armor);
    }
}

void Enemy::set_unfollowed() {
    following = false;
    for (TargetArmor &armor : armors) armor.following = false;
}

void Enemy::armor_appear(TargetArmor &) { armor_appr = true; }

void EnemyPredictorNode::update_enemy() {
    for (Enemy &enemy : enemies) {
        enemy.status = Status::Absent;
        static std::vector<int> alive_indexs(enemy.armor_cnt);
        alive_indexs.clear();

        bool enemy_kf_init_flag = false;

        for (int i = 0; i < (int)enemy.armors.size(); ++i) {
            TargetArmor &armor = enemy.armors[i];
            if (armor.status == Status::Alive) {
                enemy.status = Alive;
                enemy.alive_ts = armor.alive_ts;  // Alive的装甲板必然拥有最新的时间戳
                alive_indexs.push_back(i);
                cout << "alive" << endl;
            }
        }
        enemy.t_absent = recv_detection.time_stamp - enemy.alive_ts;
        enemy.armor_cnt = get_armor_cnt(static_cast<armor_type>(enemy.id % 9));
        // 没有检测到装甲板，不进行更新
        if (enemy.status == Status::Absent) {
            continue;
        }
        // 更新dis2d
        enemy.min_dis_2d = INFINITY;
        for (TargetArmor &armor : enemy.armors) {
            enemy.min_dis_2d = std::min(enemy.min_dis_2d, armor.dis_2d);
        }
        // for (int i = 0; i < (int)enemy.armors.size(); ++i) {
        TargetArmor &armor = enemy.armors[alive_indexs[0]];
        // if (armor.status == Status::Alive) {
        enemy_KF_4::Output now_output;
        now_output.x = armor.getpos_xyz()[0];
        now_output.y = armor.getpos_xyz()[1];
        now_output.z = armor.getpos_xyz()[2];
        now_output.Re = cos(armor.position_data.yaw);
        now_output.Im = sin(armor.position_data.yaw);
        if (!enemy.enemy_kf_init) {
            // 还没有开始跟踪，需要初始化滤波器
            // enemy_kf_init_flag = true;

            enemy_KF_4::Output now_output;
            now_output.x = armor.getpos_xyz()[0];
            now_output.y = armor.getpos_xyz()[1];
            now_output.z = armor.getpos_xyz()[2];
            now_output.Re = cos(armor.position_data.yaw);
            now_output.Im = sin(armor.position_data.yaw);
            enemy.enemy_kf.reset(now_output, armor.phase_in_enemy);
            enemy.last_update_ekf_ts = enemy.alive_ts;
            enemy.enemy_kf_init = true;
        }
        cout << "phase_id  " << armor.phase_in_enemy << endl;
        enemy.enemy_kf.CKF_update(enemy.enemy_kf.get_Z(now_output), enemy.alive_ts - enemy.last_update_ekf_ts, armor.phase_in_enemy);
        // break;
        // }
        // }

        // rviz可视化
        visualization_msgs::msg::MarkerArray marker_array;
        Eigen::Vector3d pos = enemy.enemy_kf.get_center(enemy.enemy_kf.state);
        visualization_msgs::msg::Marker marker;
        int id = 0;
        // 画中心
        marker.header.frame_id = "odom";
        marker.header.stamp = rclcpp::Node::now();
        marker.ns = "points";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = pos[0];
        marker.pose.position.y = pos[1];
        marker.pose.position.z = pos[2];
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;  // 球的大小
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.r = 1.0;  // 球的颜色
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker_array.markers.push_back(marker);

        for (int i = 0; i < enemy.armor_cnt; ++i) {
            pos = enemy.enemy_kf.get_armor(enemy.enemy_kf.state, i);
            marker.header.frame_id = "odom";
            marker.header.stamp = rclcpp::Node::now();
            marker.ns = "points";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = pos[0];
            marker.pose.position.y = pos[1];
            marker.pose.position.z = pos[2];
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.1;  // 球的大小
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;
            marker.color.r = 0.0;  // 球的颜色
            marker.color.g = 0.0;
            marker.color.b = 0.25 * (i + 1);
            marker.color.a = 1.0;
            marker_array.markers.push_back(marker);
        }
        show_enemies_pub->publish(marker_array);
        cout << "status" << armor.status << endl;
    }
}