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

IterEnemy EnemyPredictorNode::select_enemy_nearest2d() {
    IterEnemy nearest = std::min_element(enemies.begin(), enemies.end(), [&](const Enemy &a, const Enemy &b) {  // 找到二维距准星最近点
        return a.min_dis_2d < b.min_dis_2d;
    });
    if (nearest != enemies.end() && nearest->id % 9 == armor_type::TOP)  // 普通模式忽略顶装甲板
        return enemies.end();
    return nearest;
}
IterEnemy EnemyPredictorNode::select_enemy_lobshot() {
    IterEnemy top_armor = std::find_if(enemies.begin(), enemies.end(), [&](const Enemy &x) {  // 寻找敌方的顶装甲
        return x.id % 9 == armor_type::TOP;
    });
    return top_armor;
}
ballistic::bullet_res EnemyPredictorNode::calc_ballistic(const armor_EKF &armor_kf, double delay) {
    ballistic::bullet_res ball_res;
    double t_fly = 0.;  // 需迭代求出的飞行时间
    for (int i = 1; i <= 3; i++) {
        // auto xyz = pyd2xyz(armor_kf.predict(t_fly + delay));
        // auto pyd = armor_kf.predict(t_fly + delay);
        ball_res = bac->final_ballistic(pyd2xyz(armor_kf.predict(t_fly + delay)));
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
        auto predict_pos = follow->predict_positions(t_fly + delay + follow->t_absent);
        ball_res = bac->final_ballistic(predict_pos.armors[armor_phase]);
        if (ball_res.fail) {
            RCLCPP_WARN(get_logger(), "too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}
ballistic::bullet_res EnemyPredictorNode::center_ballistic(const IterEnemy &follow, double delay) {
    ballistic::bullet_res ball_res;
    double t_fly = 0;  // 飞行时间（迭代求解）
    for (int i = 0; i < 3; ++i) {
        auto predict_center = follow->predict_positions(t_fly + delay + follow->t_absent).center;
        ball_res = bac->final_ballistic(predict_center);
        if (ball_res.fail) {
            RCLCPP_WARN(get_logger(), "too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    }
    return ball_res;
}

EnemyArmor EnemyPredictorNode::select_armor_directly(const IterEnemy &follow) {
    // 直接选择最优的
    // 计算目标车体中心到base系的yaw
    // 计算大致飞行时间
    Enemy::enemy_positions pos_now = follow->get_positions();
    ballistic::bullet_res ball_estimate = bac->final_ballistic(pos_now.armors[0]);
    // 预测
    Enemy::enemy_positions pos_predict = follow->predict_positions(params.response_delay + ball_estimate.t + follow->t_absent);

    double yaw_center = atan2(pos_predict.center[1], pos_predict.center[0]);
    // 选取最正对的装甲板
    double min_dis_yaw = INFINITY;
    int min_armor_phase = -1;
    for (int i = 0; i < follow->armor_cnt; ++i) {
        double dis = abs(get_disAngle(pos_predict.armor_yaws[i], yaw_center + M_PI));  // 加PI，换方向
        if (dis < min_dis_yaw) {
            min_dis_yaw = dis;
            min_armor_phase = i;
        }
    }
    EnemyArmor res;
    res.phase = min_armor_phase;
    res.yaw_distance_predict = min_dis_yaw;
    res.pos = pos_now.armors[res.phase];
    return res;
}
TargetArmor &EnemyPredictorNode::select_armor_old(const IterEnemy &enemy) {
    using IterArmor = std::vector<TargetArmor>::iterator;

    IterArmor armor_follow = std::find_if(enemy->armors.begin(), enemy->armors.end(), [&](const TargetArmor &armor) { return armor.following; });

    if (enemy->id % 9 == armor_type::OUTPOST) {
        // TODO: 选择前哨站顶部装甲
    }

    if (enemy->is_rotate) {
        if (enemy->armor_appr) {  // 反陀螺时快速切换到新出现的装甲板
            // logger.critical("APPR");
            if (armor_follow != enemy->armors.end()) {
                armor_follow->following = false;
            }
            armor_follow = std::max_element(enemy->armors.begin(), enemy->armors.end(),
                                            [&](const TargetArmor &a, const TargetArmor &b) { return a.first_ts < b.first_ts; });
            armor_follow->following = true;
            return *armor_follow;
        } else {
            if (armor_follow == enemy->armors.end()) {
                return select_armor_directly_old(enemy);
            } else {
                return *armor_follow;
            }
        }
    }

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
    if (enemy->armors.size() == 1) {
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

ControlMsg EnemyPredictorNode::get_command() {
    // 选择enemy，保存结果为new_follow
    IterEnemy follow, new_follow, top_follow;
    follow = std::find_if(enemies.begin(), enemies.end(), [&](const Enemy &enemy) { return enemy.following; });
    new_follow = select_enemy_nearest2d();

    if (!params.choose_enemy_without_autoaim_signal && !params.right_press) {  // 松开右键，清除选择的敌人的标记，返回空指令
        if (follow != enemies.end()) follow->set_unfollowed();
        ControlMsg cmd = off_cmd;
        if (new_follow != enemies.end()) cmd.follow_id = static_cast<uint8_t>(new_follow->id % 9);
        return cmd;
    }
    if (follow != enemies.end())  // 如果上一次有选择，则使用上一次的，否则直接使用最近的
        new_follow = follow;
    if (params.rmcv_id % 9 == 1 && params.lobshot) {  // 如果在吊射模式下有目标，则优先使用吊射模式的目标
        top_follow = select_enemy_lobshot();
        if (top_follow != enemies.end()) new_follow = top_follow;
    }
    if (new_follow == enemies.end())  // 如果都没有，就返回空指令
        return off_cmd;
    RCLCPP_INFO(get_logger(), "following: %d", new_follow->id);
    new_follow->following = true;

    // 选择enemy中的装甲板
    EnemyArmor target = select_armor_directly(new_follow);  // 整车建模策略下选择的装甲板
    TargetArmor target_old = select_armor_old(new_follow);  // 老自瞄选择的装甲板

    // 根据历史信息记录yaw的最大最小值（老自瞄反陀螺需要的信息）
    double yaw_min = 0, yaw_max = 0;
    // bool mono_exist = 1;
    // 冗余处理
    if (new_follow->mono_dec.empty() || new_follow->mono_inc.empty()) {
        RCLCPP_ERROR(get_logger(), "mono empty !!!");
        // mono_exist = 0;
    } else {
        yaw_min = new_follow->mono_inc.front().second;
        yaw_max = new_follow->mono_dec.front().second;
    }
    if (params.debug) {
        cv::line(recv_detection.img, pc.pos2img(pyd2xyz(Eigen::Vector3d{0.7, yaw_min, 10.})),
                 pc.pos2img(pyd2xyz(Eigen::Vector3d{-0.7, yaw_min, 10.})), cv::Scalar(0, 0, 255));
        cv::line(recv_detection.img, pc.pos2img(pyd2xyz(Eigen::Vector3d{0.7, yaw_max, 10.})),
                 pc.pos2img(pyd2xyz(Eigen::Vector3d{-0.7, yaw_max, 10.})), cv::Scalar(0, 0, 255));
    }

    ControlMsg outpost_cmd;
    // bool use_outpost = 0;
    ballistic::bullet_res follow_ball, center_ball;
    if (new_follow->is_rotate) {
        follow_ball = calc_ballistic(new_follow, target.phase, params.response_delay);
        center_ball = center_ballistic(new_follow, params.response_delay);

        if (follow_ball.fail) return off_cmd;
        // 画一下follow_bal
        if (params.debug) {
            cv::circle(recv_detection.img, pc.pos2img(target.pos), 3, cv::Scalar(0, 0, 255), 5);
            cv::circle(show_enemies, cv::Point2d(320 - target.pos[1] * 50, 320 - target.pos[0] * 50), 2, cv::Scalar(0, 0, 255), 2);
        }
        ControlMsg cmd = make_cmd(0., (float)follow_ball.pitch, (float)follow_ball.yaw, 1, static_cast<uint8_t>(new_follow->id % 9));
        // 自动开火条件判断
        double target_dis = get_dis3d(target.pos);
        double gimbal_error_dis;
        RCLCPP_INFO(get_logger(), "yaw_spd: %lf", abs(new_follow->ekf.Xe[5]));
        if (params.rmcv_id % 9 == 1 && new_follow->id % 9 == armor_type::OUTPOST) {  // 英雄打前哨站（特化处理）
            RCLCPP_INFO(get_logger(), "OUTPOST_MODE!!!!");
            Eigen::Matrix<double, 3, 1> aiming_pos;  // pyd
            aiming_pos[0] = target_old.kf.Xe[0];     // 获取当前瞄准的定点的pitch值
            aiming_pos[1] = angle_middle(yaw_min, yaw_max);
            aiming_pos[2] = target_old.kf.Xe[2];  // 与distance值

            ballistic::bullet_res ball_res = bac->final_ballistic(pyd2xyz(aiming_pos));
            double t_flying = ball_res.t;
            outpost_cmd = make_cmd(0., (float)ball_res.pitch, (float)ball_res.yaw, 1, 6);
            if (new_follow->TSP.size() <= 3) return outpost_cmd;

            aiming_pos[2] = new_follow->common_middle_dis.get();
            // double period = 2.5/3;
            double period = (new_follow->TSP.back().first - new_follow->TSP.front().first) / (new_follow->TSP.size() - 1);
            RCLCPP_INFO(get_logger(), "period : %lf", period);
            std::pair tmp = new_follow->TSP.back();
            double pitch1 = tmp.second;
            new_follow->TSP.pop_back();
            double pitch2 = new_follow->TSP.back().second;
            // double pitch2=pitch1;
            new_follow->TSP.push_back(tmp);
            RCLCPP_INFO(get_logger(), "pitch1 pitch2 : %lf %lf", pitch1, pitch2);
            int i;
            std::queue<std::pair<double, double> > tmp_TSP;
            while (!tmp_TSP.empty()) tmp_TSP.pop();

            // 使用最小二乘法计算过中线的时间周期mini_k和最后一次过中线的时机las_middle_TS
            i = 0;
            int n = 0;
            double sumxy = 0.0, sumx = 0.0, sumy = 0.0;
            double sumx2 = 0.0;
            while (!new_follow->TSP.empty()) {
                n++;
                tmp_TSP.push(new_follow->TSP.front());
                new_follow->TSP.front().first;
                sumxy += n * new_follow->TSP.front().first;
                sumx += n;
                sumy += new_follow->TSP.front().first;
                sumx2 += n * n;
                new_follow->TSP.pop_front();
            }
            double mini_k = (sumxy - sumx * sumy / n) / (sumx2 - sumx * sumx / n);
            double mini_b = sumy / n - mini_k * sumx / n;
            double las_middle_TS = mini_k * n + mini_b;
            RCLCPP_INFO(get_logger(), "las_TS : %lf  %lf", las_middle_TS, tmp.first);
            while (!tmp_TSP.empty()) {
                new_follow->TSP.push_back(tmp_TSP.front());
                tmp_TSP.pop();
            }

            for (i = 1; i <= 2; i++) {
                if (new_follow->common_yaw_spd.get() < -AMeps)  // 向右旋转
                    aiming_pos[1] = angle_kmiddle(yaw_min, yaw_max, 0.52);
                else
                    aiming_pos[1] = angle_kmiddle(yaw_min, yaw_max, 0.48);
                if (i & 1)
                    aiming_pos[0] = pitch2;
                else
                    aiming_pos[0] = pitch1;
                ball_res = bac->final_ballistic(pyd2xyz(aiming_pos));
                t_flying = ball_res.t;
                RCLCPP_INFO(get_logger(), "t_flying: %lf", t_flying);
                if (i * period + las_middle_TS + params.timestamp_thresh > recv_detection.time_stamp + params.shoot_delay + t_flying) {
                    outpost_cmd.pitch = ball_res.pitch;
                    outpost_cmd.yaw = ball_res.yaw;
                    break;
                }
            }
            double time_error = fabs(i * period + las_middle_TS - (recv_detection.time_stamp + params.shoot_delay + t_flying));
            RCLCPP_INFO(get_logger(), "time_error: %lf", time_error);
            if (i <= 2 && time_error < params.timestamp_thresh) {
                double gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{imu.pitch, ball_res.yaw, target_old.getpos_pyd()[2]}),
                                                               pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_old.getpos_pyd()[2]}));
                if (gimbal_error_dis < params.gimbal_error_dis_thresh) outpost_cmd.flag = 2;
            }
            aiming_pos = pyd2xyz(aiming_pos);
            aiming_pos = xyz2dyz(aiming_pos);
            if (params.is_aim_top) {
                aiming_pos[2] += params.outpost_top_offset_z;
                aiming_pos[0] += params.outpost_top_offset_dis;
            }
            aiming_pos = dyz2xyz(aiming_pos);
            for (int i = 0; i < 3; ++i) {
                new_follow->outpost_aiming_pos[i].update(aiming_pos[i]);
            }
            for (int i = 0; i < 3; ++i) {
                aiming_pos[i] = new_follow->outpost_aiming_pos[i].get();
            }
            ball_res = bac->final_ballistic(aiming_pos);
            return outpost_cmd;
        } else if (new_follow->is_move) {  // 移动陀螺
            cmd.flag = 1;
        } else if (new_follow->is_high_spd_rotate && (new_follow->armor_cnt == 4)) {  // 4装甲板高速陀螺(瞄中)
            RCLCPP_INFO(get_logger(), "high_spd!!!!!!!!!!!!!!!!!!");
            gimbal_error_dis = INFINITY;
            // 在四个装甲板预测点中选一个gimbal_error_dis最小的
            Enemy::enemy_positions enemy_pos = new_follow->predict_positions(follow_ball.t + params.shoot_delay + follow->t_absent);
            for (int k = 0; k < new_follow->armor_cnt; ++k) {
                ballistic::bullet_res shoot_ball = bac->final_ballistic(enemy_pos.armors[k]);
                if (!shoot_ball.fail) {  // 对装甲板的预测点计算弹道，若成功，则更新gimbal_error_dis
                    gimbal_error_dis = std::min(gimbal_error_dis, calc_gimbal_error_dis(shoot_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
                }
            }
            if (is_big_armor(static_cast<armor_type>(new_follow->id % 9))) {  // 针对大装甲板放宽发弹阈值，增长可击打窗口期，下同
                gimbal_error_dis /= 3.0;
            }
            cmd.yaw = center_ball.yaw;
            RCLCPP_INFO(get_logger(), "min_gimbal_error_dis: %lf", gimbal_error_dis);
            // 第一条为冗余判据，保证当前解算target_dis时的装甲板较为正对，减少dis抖动，可调，下同
            if (target.yaw_distance_predict < 65.0 / 180.0 * M_PI && gimbal_error_dis < params.gimbal_error_dis_thresh) {
                cmd.flag = 2;
            } else {
                cmd.flag = 1;
            }
        } else {  // 其他的陀螺情况（跟随+预先切换）
            gimbal_error_dis = calc_surface_dis_xyz(pyd2xyz(Eigen::Vector3d{imu.pitch, follow_ball.yaw, target_dis}),
                                                    pyd2xyz(Eigen::Vector3d{imu.pitch, imu.yaw, target_dis}));
            if (is_big_armor(static_cast<armor_type>(new_follow->id % 9))) {
                gimbal_error_dis /= 3.0;
            }
            if (target.yaw_distance_predict < 65.0 / 180.0 * M_PI && gimbal_error_dis < params.gimbal_error_dis_thresh) {
                cmd.flag = 2;
            } else {
                cmd.flag = 1;
            }
        }
        return cmd;
    } else {  // 纯平移目标
        follow_ball = calc_ballistic(target_old.kf, params.response_delay);
        if (follow_ball.fail) return off_cmd;
        ControlMsg cmd = make_cmd(0., (float)follow_ball.pitch, (float)follow_ball.yaw, 1, static_cast<uint8_t>(new_follow->id % 9));
        double gimbal_error_dis = calc_gimbal_error_dis(follow_ball, Eigen::Vector3d{imu.pitch, imu.yaw, target_old.getpos_pyd()[2]});

        RCLCPP_INFO(get_logger(), "ged: %lf", gimbal_error_dis);
        double now_delay_time = 0;

        if (gimbal_error_dis < params.gimbal_error_dis_thresh) {
            // 低于某速度并且在范围内，可以使用高频射击
            if (fabs(new_follow->common_yaw_spd.get()) < params.low_spd_thresh && target_old.getpos_pyd()[2] < params.dis_thresh_kill) cmd.flag = 3;
            // 移动速度在一定范围内，可以使用普通频率射击
            else if (fabs(new_follow->common_yaw_spd.get()) < params.low_spd_thresh)
                cmd.flag = 2;
            else
                cmd.flag = 1;
        } else
            cmd.flag = 1;
        RCLCPP_INFO(get_logger(), "cmd: %lf %lf", cmd.pitch * 180.0 / M_PI, cmd.yaw * 180.0 / M_PI);
        RCLCPP_INFO(get_logger(), "imu: %lf %lf", imu.pitch * 180.0 / M_PI, imu.yaw * 180.0 / M_PI);
        return cmd;
    }
}