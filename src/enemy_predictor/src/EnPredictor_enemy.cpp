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
    bool has_alive = false;
    double nearest_ts = 0;
    double nearest_id = 0;
    for (int i = 0; i < (int)armors.size(); ++i) {
        if (armors[i].alive_ts > nearest_ts) {
            nearest_ts = armors[i].alive_ts;
            nearest_id = i;
        }
    }
    if (!has_alive && armors.size() > 0) {
        if (check_left(armor.getpos_pyd(), armors[nearest_id].getpos_pyd())) {
            armor.phase_in_enemy = (armors[nearest_id].phase_in_enemy - 1 + armor_cnt) % armor_cnt;
        } else {
            armor.phase_in_enemy = (armors[nearest_id].phase_in_enemy + 1 + armor_cnt) % armor_cnt;
        }
        has_alive = true;
    } else {
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

void Enemy::update_motion_state() {
    common_move_spd.update(get_move_spd());
    common_rotate_spd.update(get_rotate_spd());
    if (abs(common_move_spd.get()) > predictor->params.move_thresh) {
        is_move = true;
    }
    if (abs(common_rotate_spd.get()) > predictor->params.rotate_thresh) {
        is_rotate = true;
    }
    if (abs(common_rotate_spd.get()) > predictor->params.high_spd_rotate_thresh) {
        is_high_spd_rotate = true;
    }
    if (abs(common_move_spd.get()) < predictor->params.move_exit) {
        is_move = false;
    }
    if (abs(common_rotate_spd.get()) < predictor->params.rotate_exit) {
        is_rotate = false;
    }
    if (abs(common_rotate_spd.get()) < predictor->params.high_spd_rotate_exit) {
        is_high_spd_rotate = false;
    }
    if (id % 9 == armor_type::OUTPOST && (abs(common_yaw_spd.get()) > AMeps)) {
        is_rotate = true;
        is_high_spd_rotate = true;
    }
}

void Enemy::set_unfollowed() {
    following = false;
    for (TargetArmor &armor : armors) armor.following = false;
}

void Enemy::armor_appear(TargetArmor &) { armor_appr = true; }

Enemy::enemy_positions Enemy::predict_positions(double stamp) {
    enemy_positions result(4);
    enemy_KF_4::State state_pre = enemy_kf.predict(stamp);
    result.center = Eigen::Vector3d(state_pre.x, state_pre.y, enemy_kf.const_z[0]);
    // cout << "center: " << result.center << endl;
    for (int i = 0; i < armor_cnt; ++i) {
        enemy_KF_4::Output output_pre = enemy_kf.get_output(enemy_kf.h(enemy_kf.get_X(state_pre), i));
        result.armors[i] = Eigen::Vector3d(output_pre.x, output_pre.y, output_pre.z);
        result.armor_yaws[i] = output_pre.yaw + i * enemy_kf.angle_dis;
    }
    return result;
}

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
        double angle_dis = M_PI * 2 / enemy.armor_cnt;
        int big_idx, small_idx;
        if (alive_indexs.size() > 1) {
            if (enemy.armors[alive_indexs[0]].area_2d > enemy.armors[alive_indexs[1]].area_2d) {
                big_idx = 0;
                small_idx = 1;
            } else {
                big_idx = 1;
                small_idx = 0;
            }
        } else {
            big_idx = 0;
            small_idx = 0;
        }
        enemy.tracking_index = big_idx;
        TargetArmor &armor = enemy.armors[alive_indexs[big_idx]];
        // if (armor.status == Status::Alive) {
        enemy_KF_4::Output now_output;
        now_output.x = armor.getpos_xyz()[0];
        now_output.y = armor.getpos_xyz()[1];
        now_output.z = armor.getpos_xyz()[2];
        now_output.yaw = armor.position_data.yaw - armor.phase_in_enemy * angle_dis;
        if (!enemy.enemy_kf_init) {
            // 还没有开始跟踪，需要初始化滤波器
            // enemy_kf_init_flag = true;
            enemy.last_yaw = now_output.yaw;
            enemy.yaw_round = 0;
            std::vector<double> dis(4, 0.2);
            std::vector<double> z(4, now_output.z);
            enemy.enemy_kf.reset(now_output, armor.phase_in_enemy, enemy.armor_cnt, enemy.alive_ts, dis, z);
            enemy.last_update_ekf_ts = enemy.alive_ts;
            enemy.enemy_kf_init = true;
        }
        // 处理过0
        if (now_output.yaw - enemy.last_yaw < -M_PI * 1.5) {
            enemy.yaw_round++;
        } else if (now_output.yaw - enemy.last_yaw > M_PI * 1.5) {
            enemy.yaw_round--;
        }
        enemy.last_yaw = now_output.yaw;
        now_output.yaw = now_output.yaw + enemy.yaw_round * 2 * M_PI;
        cout << "yaw  " << now_output.yaw << endl;
        enemy_KF_4::Output2 now_outputs;

        if (alive_indexs.size() > 1 && enemy.armor_cnt == 4) {
            TargetArmor &armor2 = enemy.armors[alive_indexs[small_idx]];
            enemy_KF_4::Output now_output2;
            now_output2.x = armor2.getpos_xyz()[0];
            now_output2.y = armor2.getpos_xyz()[1];
            now_output2.z = armor2.getpos_xyz()[2];
            now_output2.yaw = armor2.position_data.yaw - armor2.phase_in_enemy * angle_dis;

            double theta = armor.position_data.yaw;
            double theta2 = armor2.position_data.yaw;

            double angle_diff = (acos(cos(theta2 - theta)) - M_PI / 2);
            cout << "angle_diff:  " << 1 - abs(angle_diff) << endl;

            double r = ((now_output.x - now_output2.x) * sin(theta2) - (now_output.y - now_output2.y) * cos(theta2)) / sin(theta2 - theta);
            double r2 = ((now_output2.x - now_output.x) * sin(theta) - (now_output2.y - now_output.y) * cos(theta)) / sin(theta - theta2);

            r *= 1 - abs(angle_diff);  // 没有任何道理的杂技修正
            r2 *= 1 - abs(angle_diff);

            if (r < 0.30 && r > 0.12 && r2 < 0.30 && r2 > 0.12) {
                enemy.armor_dis_filters[armor.phase_in_enemy].update(r);
                enemy.armor_dis_filters[armor2.phase_in_enemy].update(r2);
            }
        }
        enemy.armor_z_filters[armor.phase_in_enemy].update(now_output.z);
        std_msgs::msg::Float64 z_msg;

        for (int i = 0; i < enemy.enemy_kf.const_dis.size(); ++i) {
            if (enemy.armor_cnt == 4) {
                enemy.enemy_kf.const_dis[i] = enemy.armor_dis_filters[i].get();
            }
            enemy.enemy_kf.const_z[i] = enemy.armor_z_filters[i].get();
            // cout << "const" + std::to_string(i) + ": " << enemy.enemy_kf.const_dis[i] << " " << enemy.enemy_kf.const_z[i] << endl;
        }

        enemy.enemy_kf.CKF_update(enemy.enemy_kf.get_Z(now_output), enemy.alive_ts, armor.phase_in_enemy);
        Eigen::Vector3d pyd = pyd2xyz(enemy.enemy_kf.get_center(enemy.enemy_kf.state));
        enemy.ori_diff = Eigen::Vector2d(pyd[0] - imu.pitch, pyd[1] - imu.yaw).norm();

        enemy.update_motion_state();

        // rviz可视化
        // 当前滤波的中心
        // add_point_Marker(0.1, 0.1, 0.1, 1.0, 0.0, 0.0, 1.0, enemy.enemy_kf.get_center(enemy.enemy_kf.state));
        // 按phase_id:0-3颜色由浅至深显示当前滤波的装甲板
        // for (int i = 0; i < enemy.armor_cnt; ++i) {
        //     add_point_Marker(0.1, 0.1, 0.1, 0.0, 0.0, 0.25 * (i + 1), 0.5, enemy.predict_positions(enemy.alive_ts).armors[i]);
        // }
        // 观测到的装甲板位姿
        // for (int i = 0; i < alive_indexs.size(); ++i) {
        //     Eigen::Vector3d pos = enemy.armors[alive_indexs[i]].getpos_xyz();
        //     add_point_Marker(0.1, 0.1, 0.1, 0.0, 1.0, 0.0, 1.0, pos);
        //     pub_odemetry(pnp_pose_pub, pos, {0, 0, enemy.armors[alive_indexs[i]].position_data.yaw});
        // }
    }
}