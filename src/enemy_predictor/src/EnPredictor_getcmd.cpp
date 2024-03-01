#include <enemy_predictor/enemy_predictor.h>
using namespace enemy_predictor;

ControlMsg EnemyPredictorNode::make_cmd(double roll, double pitch, double yaw, uint8_t flag, uint8_t follow_id) {
    ControlMsg now;
    now.roll = (float)roll;
    now.pitch = (float)pitch;
    now.yaw = (float)yaw;
    now.flag = flag;
    now.follow_id = follow_id;
    return now;
}

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
    for (int i = 0; i < follow->armor_cnt; ++i) {
        double dis = abs(get_disAngle(pos_predict.armor_yaws[i], yaw_center + M_PI));  // 加PI，换方向
        show_data.data = dis / M_PI * 180;
        watch_data_pubs[i]->publish(show_data);
        if (dis < min_dis_yaw) {
            min_dis_yaw = dis;
            selected_id = i;
        }
    }
    if (recv_detection.time_stamp - change_target_armor_ts < params.change_armor_time_thresh) {
        selected_id = last_selected_id;
        min_dis_yaw = abs(get_disAngle(pos_predict.armor_yaws[selected_id], yaw_center + M_PI));
    }
    if (selected_id != last_selected_id) {
        change_target_armor_ts = recv_detection.time_stamp;
    }
    EnemyArmor res;
    show_data.data = change_target_armor_ts;
    watch_data_pubs[4]->publish(show_data);

    res.phase = selected_id;
    res.yaw_distance_predict = min_dis_yaw;
    res.pos = pos_now.armors[res.phase];
    last_selected_id = selected_id;

    // if (abs(get_disAngle(min_dis_yaw, last_min_dis_yaw)) > 40 / 180 * M_PI) {
    //     res.phase = selected_id;
    //     res.yaw_distance_predict = min_dis_yaw;
    //     res.pos = pos_now.armors[res.phase];
    //     last_selected_id = selected_id;
    // } else {
    //     res.phase = last_selected_id;
    //     res.yaw_distance_predict = last_min_dis_yaw;
    //     res.pos = pos_now.armors[res.phase];
    // }

    return res;
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
            cmd.follow_id = static_cast<uint8_t>(new_follow->id % 9);
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

    ballistic::bullet_res follow_ball, center_ball;
    follow_ball = calc_ballistic(new_follow, target.phase, params.response_delay);
    center_ball = center_ballistic(new_follow, params.response_delay);
    if (follow_ball.fail) {
        return off_cmd;
    }
    ControlMsg cmd = make_cmd(0., (float)follow_ball.pitch, (float)follow_ball.yaw, 1, static_cast<uint8_t>(new_follow->id % 9));
    // 自动开火条件判断
    double target_dis = get_dis3d(target.pos);
    double gimbal_error_dis;
    if (new_follow->is_high_spd_rotate) {
        RCLCPP_INFO(get_logger(), "high_spd!!!!!!!!!!!!!!!!!!");
        gimbal_error_dis = INFINITY;
        // 在四个装甲板预测点中选一个gimbal_error_dis最小的
        Enemy::enemy_positions enemy_pos = new_follow->predict_positions(recv_detection.time_stamp + follow_ball.t + params.shoot_delay);
        for (int k = 0; k < new_follow->armor_cnt; ++k) {
            ballistic::bullet_res shoot_ball = bac->final_ballistic(enemy_pos.armors[k]);
            if (!shoot_ball.fail) {  // 对装甲板的预测点计算弹道，若成功，则更新gimbal_error_dis
                gimbal_error_dis = std::min(gimbal_error_dis, calc_gimbal_error_dis(shoot_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
            }
        }
        cmd.yaw = center_ball.yaw;
        RCLCPP_INFO(get_logger(), "min_gimbal_error_dis: %lf", gimbal_error_dis);
        // 第一条为冗余判据(?)，保证当前解算target_dis时的装甲板较为正对，减少dis抖动，可调，下同
        if (target.yaw_distance_predict < 35.0 / 180.0 * M_PI && gimbal_error_dis < params.gimbal_error_dis_thresh) {
            cmd.flag = 3;
        } else {
            cmd.flag = 1;
        }
    } else {
        gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{imu.pitch, follow_ball.yaw, target_dis}),
                                                pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
        if (target.yaw_distance_predict < 60.0 / 180.0 * M_PI && gimbal_error_dis < params.gimbal_error_dis_thresh) {
            cmd.flag = 3;
        } else {
            cmd.flag = 1;
        }
    }
    // pub目标和预测
    add_point_Marker(0.1, 0.1, 0.1, 0, 1, 1, 1, target.pos);
    add_point_Marker(0.1, 0.1, 0.1, 1, 1, 0, 1, new_follow->predict_positions(follow_ball.t + recv_detection.time_stamp).armors[target.phase]);
    return cmd;
}