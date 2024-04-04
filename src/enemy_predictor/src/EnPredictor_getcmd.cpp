#include <enemy_predictor/enemy_predictor.h>
using namespace enemy_predictor;

IterEnemy EnemyPredictorNode::select_enemy_oritation() {
    IterEnemy nearest = std::min_element(enemies.begin(), enemies.end(), [&](const Enemy &a, const Enemy &b) {  // 找到二维距准星最近点
        return a.ori_diff < b.ori_diff;
    });
    if (nearest != enemies.end() && nearest->id % 9 == armor_type::TOP)  // 普通模式忽略顶装甲板
    {
        return enemies.end();
    }
    // cout << "id" << nearest->id << endl;
    return nearest;
}

EnemyArmor EnemyPredictorNode::select_armor_directly(const IterEnemy &follow) {
    // 直接选择最优的
    // 计算大致飞行时间
    Enemy::enemy_positions pos_now = follow->predict_positions(recv_detection.time_stamp);
    ballistic::bullet_res ball_estimate = bac->final_ballistic(pos_now.center);
    // 预测
    Enemy::enemy_positions pos_predict = follow->predict_positions(params.response_delay + ball_estimate.t + recv_detection.time_stamp);

    double yaw_center = atan2(pos_predict.center[1], pos_predict.center[0]);
    // 选取最正对的装甲板
    double min_dis_yaw = INFINITY;
    // double last_min_dis_yaw = abs(get_disAngle(pos_predict.armor_yaws[last_min_dis_yaw], yaw_center + M_PI));
    // int min_armor_phase = -1;
    int selected_id = -1;
    is_change_target_armor = false;
    std_msgs::msg::Float64 show_data;
    // TODO: 统计同一phase_id的装甲板yaw角出现范围，能否过PI,判断可能是不转，或者是有障碍物，
    // 通过速度、yaw角出现特定范围的频率进一步判定
    for (int i = 0; i < follow->armors.size(); ++i) {
        double now_dis_yaw = get_disAngle(follow->armors[i].yaw_kf.Xe[0], yaw_center + M_PI);
        follow->dis_yaw_queue.push(now_dis_yaw);
        follow->dis_yaw_set.insert(now_dis_yaw);
        if (follow->dis_yaw_queue.size() > follow->max_yaw_num) {
            double old_num = follow->dis_yaw_queue.front();
            follow->dis_yaw_queue.pop();
            follow->dis_yaw_set.erase(follow->dis_yaw_set.find(old_num));
        }
        // int phase_id = follow->armors[i].phase_in_enemy;
        // auto yaw_history = follow->armors_yaw_history[phase_id];
        // yaw_history.push_back(now_dis_yaw);
        // pub_odemetry(armor_yaw_pubs[phase_id], {0, 0, 0}, {0, 0, now_dis_yaw});

        // if (yaw_history.size() > 100) {
        //     yaw_history.erase(yaw_history.begin());
        // }
        // foxglove_pub(watch_data_pubs[phase_id], sin(now_dis_yaw));
        // follow->armor_disyaw_mean_filters[phase_id].update(now_dis_yaw);
        // follow->armor_disyaw_mean2_filters[phase_id].update(now_dis_yaw * now_dis_yaw);
        // follow->center_pos_history.push_back(follow->enemy_kf.get_center(follow->enemy_kf.state).block(0,0,2,0));
    }

    // 不动或者自动跟踪平动的目标
    // 不打被挡住的，装甲板的区域有多少置于其中。
    double max_size = 0;
    follow->max_yaw_llimit = INFINITY;
    follow->max_yaw_rlimit = -INFINITY;

    for (int i = 0; i < follow->armor_cnt; ++i) {
        if (follow->armors_yaw_history[i].size() > max_size) {
            max_size = follow->armors_yaw_history[i].size();
        }
    }

    // for (int i = 0; i < follow->armor_cnt; ++i) {
    //     double mean = follow->armor_disyaw_mean_filters[i].get();
    //     double var = follow->armor_disyaw_mean2_filters[i].get() - mean * mean;
    //     // 最大的左/右视差角，假设均匀分布
    //     double yaw_llimit = mean - sqrt(3 * var);
    //     double yaw_rlimit = mean + sqrt(3 * var);
    //     follow->armors_disyaw_llimit[i] = yaw_llimit;
    //     follow->armors_disyaw_rlimit[i] = yaw_rlimit;
    //     if (yaw_llimit < follow->max_yaw_llimit) {
    //         follow->max_yaw_llimit = yaw_llimit;
    //     }
    //     if (yaw_rlimit > follow->max_yaw_rlimit) {
    //         follow->max_yaw_rlimit = yaw_rlimit;
    //     }
    //     pub_pose(armor_disyaw_llimit_pubs[i], {0, 0, 1}, {0, 0, yaw_llimit});
    //     pub_pose(armor_disyaw_rlimit_pubs[i], {0, 0, -1}, {0, 0, yaw_rlimit});
    // }

    // if (fabs(max_yaw_llimit) < M_PI / 4. * (7. / 8)) {
    //     follow->wall_left_hidden = true;
    // } else {
    //     follow->wall_left_hidden = false;
    // }
    // if (fabs(max_yaw_rlimit) < M_PI / 4. * (7. / 8)) {
    //     follow->wall_right_hidden = true;
    // } else {
    //     follow->wall_right_hidden = false;
    // }

    for (int i = 0; i < follow->armor_cnt; ++i) {
        double pre_dis = get_disAngle(pos_predict.armor_yaws[i], yaw_center + M_PI);  // 加PI，换方向
        show_data.data = pre_dis / M_PI * 180;
        if (abs(pre_dis) < abs(min_dis_yaw)) {
            min_dis_yaw = pre_dis;
            selected_id = i;
        }

        // double static_thresh;
        // double wall_thresh;
        // bool wall_hidden = false;
        // if (fabs(follow->armors_disyaw_llimit[i] - follow->armors_disyaw_rlimit[i]) > static_thresh &&
        //     fabs(follow->armors_disyaw_llimit[i] - follow->armors_disyaw_rlimit[i]) < wall_thresh &&
        //     (fabs(follow->get_move_spd()) < 1. || fabs(follow->get_rotate_spd()) > 1.)) {
        //     wall_hidden = true;
        // }

        // if (((wall_hidden && pre_dis > follow->armors_disyaw_llimit[i] && pre_dis < follow->armors_disyaw_rlimit[i]) || !wall_hidden) &&
        //     pre_dis < min_dis_yaw) {
        //     min_dis_yaw = pre_dis;
        //     selected_id = i;
        // }
    }

    if (recv_detection.time_stamp - change_target_armor_ts < params.change_armor_time_thresh) {
        selected_id = last_selected_id;
        min_dis_yaw = get_disAngle(pos_predict.armor_yaws[selected_id], yaw_center + M_PI);
    }
    if (selected_id != last_selected_id) {
        change_target_armor_ts = recv_detection.time_stamp;
    }
    EnemyArmor res;
    show_data.data = change_target_armor_ts;
    // watch_data_pubs[4]->publish(show_data);

    res.phase = selected_id;
    res.yaw_distance_predict = min_dis_yaw;
    res.pos = pos_now.armors[res.phase];
    last_selected_id = selected_id;

    return res;
}
TargetArmor &EnemyPredictorNode::select_armor_old(const IterEnemy &enemy) {
    using IterArmor = std::vector<TargetArmor>::iterator;

    IterArmor armor_follow = std::find_if(enemy->armors.begin(), enemy->armors.end(), [&](const TargetArmor &armor) { return armor.following; });

    // 非反陀螺模式
    if (armor_follow == enemy->armors.end()) {
        return select_armor_directly_old(enemy);
    }
    if (enemy->armors.size() < 2) {  // 仅有一个装甲板
        return *armor_follow;
    }
    // 有两个装甲板
    IterArmor another = enemy->armors.begin();
    if (another == armor_follow) {
        ++another;  // 选择另一个armor
    }
    if (another->area_2d / armor_follow->area_2d > params.size_ratio_thresh) {  // 超过切换阈值，切换目标
        armor_follow->following = false;
        another->following = true;
        return *another;
    }
    return *armor_follow;
}
TargetArmor &EnemyPredictorNode::select_armor_directly_old(const IterEnemy &enemy) {
    if (enemy->armors.size() == 1) {  // 只有一个就直接选择
        enemy->armors[0].following = true;
        return enemy->armors[0];
    }
    // 有两个armor
    TargetArmor &left = enemy->armors[0], &right = enemy->armors[1];
    if (left.status == right.status) {  // 同时alive或者同时absent，选择面积大的
        TargetArmor &chosen = left.area_2d < right.area_2d ? right : left;
        chosen.following = true;
        return chosen;
    }
    TargetArmor &chosen = left.status == Status::Alive ? left : right;  // 有且仅有一个absent，选择alive的
    chosen.following = true;
    return chosen;
}
ballistic::bullet_res EnemyPredictorNode::center_ballistic(const IterEnemy &follow, double delay) {
    ballistic::bullet_res ball_res;
    double t_fly = 0;  // 飞行时间（迭代求解）
    for (int i = 0; i < 3; ++i) {
        auto predict_center = follow->predict_positions(recv_detection.time_stamp + t_fly + delay).center;
        ball_res = bac->final_ballistic(predict_center);
        if (ball_res.fail) {
            RCLCPP_WARN(get_logger(), "too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}

ballistic::bullet_res EnemyPredictorNode::calc_ballistic(const IterEnemy &follow, int armor_phase, double delay) {
    ballistic::bullet_res ball_res;
    double t_fly = 0;  // 飞行时间（迭代求解）
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d predict_pos = follow->predict_positions(recv_detection.time_stamp + t_fly + delay).armors[armor_phase];

        ball_res = bac->final_ballistic(predict_pos);
        if (ball_res.fail) {
            RCLCPP_WARN(get_logger(), "too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}
ballistic::bullet_res EnemyPredictorNode::calc_ballistic(const armor_EKF &armor_kf, double delay) {
    ballistic::bullet_res ball_res;
    double t_fly = 0.;  // 需迭代求出的飞行时间
    for (int i = 1; i <= 3; i++) {
        // auto xyz = pyd2xyz(armor_kf.predict(t_fly + delay));
        // auto pyd = armor_kf.predict(t_fly + delay);
        ball_res = bac->final_ballistic(pyd2xyz(armor_kf.predict(recv_detection.time_stamp + t_fly + delay)));
        if (ball_res.fail) {
            RCLCPP_WARN(get_logger(), "too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    }

    return ball_res;
}

ControlMsg EnemyPredictorNode::get_command() {
    // 选择enemy，保存结果为new_follow
    IterEnemy follow, new_follow, top_follow;
    follow = std::find_if(enemies.begin(), enemies.end(), [&](const Enemy &enemy) { return enemy.following; });
    new_follow = select_enemy_oritation();

    if (!params.choose_enemy_without_autoaim_signal && !params.right_press) {  // 松开右键，清除选择的敌人的标记，返回空指令
        if (follow != enemies.end()) {
            follow->set_unfollowed();
        }
        ControlMsg cmd = off_cmd;
        if (new_follow != enemies.end()) {
            cmd.vision_follow_id = static_cast<uint8_t>(new_follow->id % 9);
        }
        return cmd;
    }

    if (follow != enemies.end()) {  // 如果上一次有选择，则使用上一次的，否则直接使用最近的
        new_follow = follow;
    }

    // TODO:吊射模式

    if (new_follow == enemies.end()) {  // 如果都没有，就返回空指令
                return off_cmd;
    }

    RCLCPP_INFO(get_logger(), "following: %d", new_follow->id);
    new_follow->following = true;

    // 选择enemy中的装甲板
    EnemyArmor target = select_armor_directly(new_follow);  // 整车建模策略下选择的装甲板
    TargetArmor target_old = select_armor_old(new_follow);  // 老自瞄选择的装甲板

    ballistic::bullet_res follow_ball, center_ball;
    follow_ball = calc_ballistic(new_follow, target.phase, params.response_delay);
    center_ball = center_ballistic(new_follow, params.response_delay);
    if (follow_ball.fail) {
                return off_cmd;
    }
    send_msg shoot_behavior((float)follow_ball.pitch, (float)follow_ball.yaw, 1, 3, 20, static_cast<uint8_t>(new_follow->id % 9));
    ControlMsg cmd = make_cmd(shoot_behavior);
    // 自动开火条件判断
    // min_dis_yaw to 碰墙，墙附近反复来回，打到墙，判断最佳角度，

    double target_dis = get_dis3d(target.pos);
    double gimbal_error_dis;
    if (new_follow->is_rotate) {
        if (new_follow->is_high_spd_rotate) {
            RCLCPP_INFO(get_logger(), "high_spd!!!!!!!!!!!!!!!!!!");
            gimbal_error_dis = INFINITY;
            // 在四个装甲板预测点中选一个gimbal_error_dis最小的
            Enemy::enemy_positions enemy_pos = new_follow->predict_positions(recv_detection.time_stamp + follow_ball.t + params.shoot_delay);
            int high_target_idx;
            for (int k = 0; k < new_follow->armor_cnt; ++k) {
                ballistic::bullet_res shoot_ball = bac->final_ballistic(enemy_pos.armors[k]);
                if (!shoot_ball.fail) {  // 对装甲板的预测点计算弹道，若成功，则更新gimbal_error_dis
                    if (gimbal_error_dis > calc_gimbal_error_dis(shoot_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_dis})) {
                        gimbal_error_dis = calc_gimbal_error_dis(shoot_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_dis});
                        high_target_idx = k;
                    }
                    // gimbal_error_dis = std::min(gimbal_error_dis, calc_gimbal_error_dis(shoot_ball, Eigen::Vector3d{imu.pitch, imu.yaw,
                    // target_dis}));
                }
            }
            cmd.yaw = center_ball.yaw;
            RCLCPP_INFO(get_logger(), "min_gimbal_error_dis: %lf", gimbal_error_dis);
            if (gimbal_error_dis < params.gimbal_error_dis_thresh) {
                // cmd.flag = 3;
                cmd.rate = 20;
                cmd.one_shot_num = 1;
            } else {
                // cmd.flag = 1;
                cmd.rate = 0;
                cmd.one_shot_num = 0;
            }
        } else {
            gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{imu.pitch, follow_ball.yaw, target_dis}),
                                                    pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
            cout << "yaw_dis" << target.yaw_distance_predict << endl;
            cout << "lyaw_limit: " << *new_follow->dis_yaw_set.begin() << "ryaw_limit:" << *new_follow->dis_yaw_set.rbegin() << endl;
            if (abs(target.yaw_distance_predict) < 60.0 / 180.0 * M_PI && gimbal_error_dis < params.gimbal_error_dis_thresh &&
                target.yaw_distance_predict > *new_follow->dis_yaw_set.begin() && target.yaw_distance_predict < *new_follow->dis_yaw_set.rbegin()) {
                // cmd.flag = 3;
                cmd.rate = 20;
                cmd.one_shot_num = 3;
            } else {
                // cmd.flag = 1;
                cmd.rate = 0;
                cmd.one_shot_num = 0;
            }
        }
    } else {  // 纯平移目标
        follow_ball = calc_ballistic(target_old.kf, params.response_delay);
        if (follow_ball.fail) {
            return off_cmd;
        }
        send_msg shoot_behavior((float)follow_ball.pitch, (float)follow_ball.yaw, 1, 3, 20, static_cast<uint8_t>(new_follow->id % 9));
        cmd = make_cmd(shoot_behavior);
        double gimbal_error_dis = calc_gimbal_error_dis(follow_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_old.getpos_pyd()[2]});

        RCLCPP_INFO(get_logger(), "ged: %lf", gimbal_error_dis);

        if (gimbal_error_dis < params.gimbal_error_dis_thresh) {
            // // 低于某速度并且在范围内，可以使用高频射击
            // if (fabs(new_follow->common_yaw_spd.get()) < params.low_spd_thresh && target_old.getpos_pyd()[2] < params.dis_thresh_kill) {
            //     cmd.flag = 3;
            // }
            // // 移动速度在一定范围内，可以使用普通频率射击
            // else if (fabs(new_follow->common_yaw_spd.get()) < params.low_spd_thresh) {
            //     cmd.flag = 2;
            // } else {
            //     cmd.flag = 1;
            // }
            // cmd.flag = 3;
            cmd.rate = 20;
            cmd.one_shot_num = 1;
        } else {
            // cmd.flag = 1;
            cmd.rate = 10;
            cmd.one_shot_num = 1;
        }
        RCLCPP_INFO(get_logger(), "cmd: %lf %lf", cmd.pitch * 180.0 / M_PI, cmd.yaw * 180.0 / M_PI);
        RCLCPP_INFO(get_logger(), "imu: %lf %lf", imu.pitch * 180.0 / M_PI, imu.yaw * 180.0 / M_PI);
    }
    // pub目标和预测
    // add_point_Marker(0.1, 0.1, 0.1, 0, 1, 1, 1, target.pos);
    // add_point_Marker(0.1, 0.1, 0.1, 1, 1, 0, 1, new_follow->predict_positions(follow_ball.t + recv_detection.time_stamp).armors[target.phase]);
    return cmd;
}