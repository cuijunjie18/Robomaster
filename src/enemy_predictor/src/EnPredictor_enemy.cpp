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
        if (armors[i].tracking_in_enemy) {
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

// 获取当前坐标到敌人中心的二维平面距离
double Enemy::get_distance() { return get_dis3d(get_positions().center); }

void Enemy::armor_appear(TargetArmor &) { armor_appr = true; }

Enemy::enemy_positions Enemy::extract_from_Xe(const enemy_half_observer_EKF::Vn &_xe, double last_r, double dz) {
    enemy_positions result;
    // center
    result.center[0] = _xe[0];
    result.center[1] = _xe[2];
    result.center[2] = 0;

    // armors;
    for (int i = 0; i < armor_cnt; ++i) {
        double r = _xe[8], z = _xe[6];
        if (armor_cnt == 4 && i & 1) {
            r = last_r;
            z = _xe[6] + dz;
        }
        // 逆时针
        double now_yaw = _xe[4] + M_PI * 2 * i / armor_cnt;
        result.armor_yaws[i] = now_yaw;
        result.armors[i][0] = result.center[0] + r * cos(now_yaw);
        result.armors[i][1] = result.center[1] + r * sin(now_yaw);
        result.armors[i][2] = z;
    }
    return result;
}

Enemy::enemy_positions Enemy::get_positions() { return extract_from_Xe(ekf.Xe, ekf.last_r, dz); }

Enemy::enemy_positions Enemy::predict_positions(double dT) { return extract_from_Xe(ekf.predict(dT), ekf.last_r, dz); }
void Enemy::update_motion_state() {
    common_move_spd.update(ekf.get_move_spd());
    common_rotate_spd.update(ekf.get_rotate_spd());
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

void EnemyPredictorNode::update_enemy() {
    // static Logger add_armor_logger("Enemy_predictor:update_enemy");
    for (Enemy &enemy : enemies) {
        // 统计enemy的alive状态，先置为Absent
        enemy.status = Status::Absent;
        static std::vector<int> alive_indexs(enemy.armor_cnt);
        alive_indexs.clear();
        int tracking_armor_id = -1;
        bool tracking_absent_flag = false;  // 目前正在跟踪的装甲板暂时没出现
        bool ekf_init_flag = false;
        bool tracking_change_flag = false;
        for (int i = 0; i < (int)enemy.armors.size(); ++i) {
            TargetArmor &armor = enemy.armors[i];
            if (armor.status == Status::Alive) {
                enemy.status = Alive;
                enemy.alive_ts = armor.alive_ts;  // Alive的装甲板必然拥有最新的时间戳
                while (!enemy.mono_inc.empty() && enemy.mono_inc.back().second >= armor.get_yaw()) enemy.mono_inc.pop_back();
                enemy.mono_inc.push_back(std::make_pair(armor.alive_ts, armor.get_yaw()));
                while (!enemy.mono_dec.empty() && enemy.mono_dec.back().second <= armor.get_yaw()) enemy.mono_dec.pop_back();
                enemy.mono_dec.push_back(std::make_pair(armor.alive_ts, armor.get_yaw()));
                alive_indexs.push_back(i);
            }
            if (armor.tracking_in_enemy) {
                if (tracking_armor_id == -1) {
                    tracking_armor_id = i;
                    if (armor.status == Status::Absent) {
                        tracking_absent_flag = true;
                    }
                } else {
                    RCLCPP_ERROR(get_logger(), "[update_enemy] impossible muliple tracking armor in one Enemy!");
                }
            }
        }
        enemy.t_absent = recv_detection.time_stamp - enemy.alive_ts;
        enemy.armor_cnt = get_armor_cnt(static_cast<armor_type>(enemy.id % 9));

        RCLCPP_INFO(get_logger(), "[update_enemy] tracking_armor_id: %d", tracking_armor_id);
        // 没有检测到装甲板，不进行更新
        if (enemy.status == Status::Absent) {
            continue;
        }
        // 更新dis2d
        enemy.min_dis_2d = INFINITY;
        for (TargetArmor &armor : enemy.armors) {
            enemy.min_dis_2d = std::min(enemy.min_dis_2d, armor.dis_2d);
        }
        if (tracking_armor_id == -1 && !enemy.enemy_ekf_init) {
            // 还没有开始跟踪，需要初始化滤波器
            ekf_init_flag = true;
        }
        if (tracking_armor_id == -1 || tracking_absent_flag) {
            // 还没有跟踪的装甲板或跟踪的装甲板离线，从alive中挑选一个面积大的
            tracking_change_flag = true;  // 无论是离线还是没有都可以认为发生了装甲板跳动
            if (alive_indexs.size() == 1) {
                tracking_armor_id = alive_indexs[0];
            } else if (alive_indexs.size() == 2) {
                // 从alive中挑选一个面积大的
                if (enemy.armors[alive_indexs[0]].area_2d > enemy.armors[alive_indexs[1]].area_2d) {
                    tracking_armor_id = alive_indexs[0];
                } else {
                    tracking_armor_id = alive_indexs[1];
                }
            } else {
                RCLCPP_ERROR(get_logger(), "[update_enemy] impossible alive armor in one Enemy!");
            }
        } else {
            // 进入这里意味着之前的跟踪装甲现在也在线，所以不用考虑只有一个在线的情况，跟踪状态没有改变
            if (alive_indexs.size() == 2) {
                // 找到另一块装甲板
                int another_armor_id = alive_indexs[0] == tracking_armor_id ? alive_indexs[1] : alive_indexs[0];
                // 根据面积阈值来判断是否需要切换观测
                if (enemy.armors[another_armor_id].area_2d / enemy.armors[tracking_armor_id].area_2d > params.size_ratio_thresh) {
                    // 进行切换
                    tracking_armor_id = another_armor_id;
                    tracking_change_flag = true;
                }
            }
        }
        if (tracking_armor_id == -1) {
            RCLCPP_ERROR(get_logger(), "[update_enemy] impossible tracking armor id!");
            continue;
        }
        // 设置tracking状态
        enemy.armors[tracking_armor_id].tracking_in_enemy = true;
        for (int i = 0; i < (int)enemy.armors.size(); ++i) {
            if (i != tracking_armor_id) {
                enemy.armors[i].tracking_in_enemy = false;
            }
        }
        if (params.debug) {
            // 显示一下法向量
            cv::Point2d show_pt_st = pc.pos2img(enemy.armors[tracking_armor_id].getpos_xyz());
            cv::Point2d show_pt_end =
                pc.pos2img(enemy.armors[tracking_armor_id].getpos_xyz() + enemy.armors[tracking_armor_id].position_data.show_vec);
            cv::line(recv_detection.img, show_pt_st, show_pt_end, cv::Scalar(255, 0, 0), 2);
        }
        // logger.info("armor_jump: {} {} {}",(int)tracking_change_flag,(int)tracking_change_flag,(int)tracking_change_flag);
        // 开始更新滤波

        // 计算enemy-yaw值
        TargetArmor &tracking_armor = enemy.armors[tracking_armor_id];
        double armor_enemy_yaw = atan2(tracking_armor.position_data.normal_vec[1], tracking_armor.position_data.normal_vec[0]);

        if (ekf_init_flag || tracking_change_flag) {
            enemy.last_yaw = armor_enemy_yaw;
            enemy.yaw_round = 0;
            RCLCPP_WARN(get_logger(), "[update_enemy] armor_jump!");
        }
        // 处理过0
        if (armor_enemy_yaw - enemy.last_yaw < -M_PI * 1.5) {
            enemy.yaw_round++;
        } else if (armor_enemy_yaw - enemy.last_yaw > M_PI * 1.5) {
            enemy.yaw_round--;
        }
        enemy.last_yaw = armor_enemy_yaw;
        enemy.yaw = armor_enemy_yaw + enemy.yaw_round * 2 * M_PI;

        enemy_half_observer_EKF::Vm now_observe;
        now_observe << tracking_armor.getpos_xyz()[0], tracking_armor.getpos_xyz()[1], tracking_armor.getpos_xyz()[2], enemy.yaw;
        if (tracking_change_flag) {
            enemy.ekf.state.yaw = enemy.yaw;
            if (enemy.armor_cnt == 4) {
                // 除非是4装甲板，否则不需要两个r/z
                enemy.dz = enemy.ekf.state.z - now_observe[2];
                enemy.ekf.state.z = now_observe[2];
                std::swap(enemy.ekf.state.r, enemy.ekf.last_r);
            }
        }
        // logger.info("enemy_yaw: {} {} {}",enemy.yaw * 180 / M_PI,enemy.yaw * 180 / M_PI,enemy.yaw * 180 / M_PI);
        // logger.info("match_id: {} {} {}",tracking_armor_id,tracking_armor_id,tracking_armor_id);
        // cv::line(show_enemies, cv::Point2d(320, 320), cv::Point2d(320 - tracking_armor.position_data.normal_vec[1] * 100, 320 -
        // tracking_armor.position_data.normal_vec[0] * 100), cv::Scalar(255, 0, 0), 2);

        if (ekf_init_flag) {
            enemy.enemy_ekf_init = true;
            enemy.ekf.reset(now_observe);
            enemy.dz = 0;
            enemy.last_update_ekf_ts = enemy.alive_ts;
            RCLCPP_WARN(get_logger(), "armor_init! %ld", enemy.armors.size());
        } else {
            enemy.ekf.CKF_update(now_observe, enemy.alive_ts - enemy.last_update_ekf_ts);

            // std_msgs::msg::Float64 x_msg, y_msg;
            // x_msg.data = enemy.ekf.state.x;
            // y_msg.data = enemy.ekf.state.y;
            // watch_data_pubs[0]->publish(x_msg);
            // watch_data_pubs[1]->publish(y_msg);

            enemy.last_update_ekf_ts = enemy.alive_ts;
        }
        if (params.debug) {
            show_enemies = cv::Mat(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
            cv::circle(show_enemies, cv::Point2d(320, 320), 4, cv::Scalar(211, 0, 148), 4);
            Enemy::enemy_positions pos = enemy.get_positions();
            cv::circle(show_enemies, cv::Point2d(320 - pos.center[1] * 50, 320 - pos.center[0] * 50), 2, cv::Scalar(0, 0, 255), 2);
            cv::circle(show_enemies, cv::Point2d(320 - pos.armors[0][1] * 50, 320 - pos.armors[0][0] * 50), 2, cv::Scalar(211, 0, 148), 2);
            for (int i = 1; i < enemy.armor_cnt; ++i) {
                cv::circle(show_enemies, cv::Point2d(320 - pos.armors[i][1] * 50, 320 - pos.armors[i][0] * 50), 2, cv::Scalar(128, 128, 128), 2);
            }
            // 画观测点
            cv::circle(show_enemies, cv::Point2d(320 - now_observe[1] * 50, 320 - now_observe[0] * 50), 3, cv::Scalar(211, 0, 148), 1);

            // Enemy::enemy_positions pos_predict = enemy.predict_positions(response_delay);
            // 反投影预测点到图像
            cv::circle(recv_detection.img, pc.pos2img(pos.armors[0]), 3, cv::Scalar(211, 0, 148), 5);
            for (int i = 1; i < enemy.armor_cnt; ++i) {
                cv::circle(recv_detection.img, pc.pos2img(pos.armors[i]), 3, cv::Scalar(0, 255, 0), 5);
            }
            // 画当前法向量
            cv::line(show_enemies, cv::Point2d(320, 320),
                     cv::Point2d(320 - tracking_armor.position_data.normal_vec[1] * 50, 320 - tracking_armor.position_data.normal_vec[0] * 50),
                     cv::Scalar(255, 0, 0), 2);
            // 画当前phase
            cv::putText(show_enemies, "Phase:" + std::to_string(tracking_armor.phase_in_enemy), cv::Point2d(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.5,
                        cv::Scalar(0, 255, 0));
        }
        double census_period = std::min(params.census_period_max, std::max(params.census_period_min, enemy.appr_period * 4.0));
        if (enemy.id == armor_type::OUTPOST) {
            census_period = params.census_period_max;
        }

        for (; !enemy.mono_inc.empty() && recv_detection.time_stamp - enemy.mono_inc.front().first > census_period; enemy.mono_inc.pop_front())
            ;
        for (; !enemy.mono_dec.empty() && recv_detection.time_stamp - enemy.mono_dec.front().first > census_period; enemy.mono_dec.pop_front())
            ;
        for (;
             !enemy.TSP.empty() && (recv_detection.time_stamp - enemy.TSP.front().first > params.anti_outpost_census_period || enemy.TSP.size() > 10);
             enemy.TSP.pop_front())
            ;
        enemy.update_motion_state();
        // 应该是fg未使用的，问问qzz
        // enemy_positions_pub.push(enemy.get_positions());
    }
}
