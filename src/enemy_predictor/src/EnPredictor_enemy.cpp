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

// 获取当前坐标到敌人中心的二维平面距离
double Enemy::get_distance() { return get_dis3d(get_positions().center); }

void Enemy::armor_appear(TargetArmor &) { armor_appr = true; }

Enemy::enemy_positions Enemy::extract_from_state(const enemy_double_observer_EKF::State &state) {
    enemy_positions result;
    // center
    result.center[0] = state.x;
    result.center[1] = state.y;
    if (!tracking_absent_flag) {
        result.center[2] = state.z;
    } else {
        result.center[2] = state.z2;
    }

    // armors;
    for (int i = 0; i < armor_cnt; ++i) {
        double r, z, now_yaw;
        if (armor_cnt == 4 && i & 1) {
            r = state.r2;
            z = state.z2;
            now_yaw = state.yaw2;
        } else {
            r = state.r;
            z = state.z;
            now_yaw = state.yaw;
        }
        // 逆时针
        result.armor_yaws[i] = now_yaw;
        result.armors[i][0] = result.center[0] + r * cos(now_yaw);
        result.armors[i][1] = result.center[1] + r * sin(now_yaw);
        result.armors[i][2] = z;
    }
    return result;
}

Enemy::enemy_positions Enemy::get_positions() { return extract_from_state(ekf.state); }

Enemy::enemy_positions Enemy::predict_positions(double dT) { return extract_from_state(ekf.predict(dT)); }
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

void Enemy::observe_filter(std::vector<double> &data, double &sample, const int& method, bool in_scope) {
    // 数据粗筛+平均
    // method:
    //  1.算术平均
    //  2.几何平均
    //  3.调和平均
    // 也可调整观测噪声矩阵
    if (in_scope) {
        data.push_back(sample);
        if (data.size() > 100) {
            data.erase(data.begin());
        }
    }
    if (data.size() > 1) {
        double mean = (method == 2) ? 1 : 0;
        double inv_var = 0;
        for (double &sample_ : data) {
            // 几何、调和平均值受偏离过大的测量值影响小
            switch (method)
            {
            case 2:
                mean *= sample_;
                break;
            case 3:
                mean += 1./sample_;
                break;
            case 1:
            default:
                mean += sample_;
                break;
            }
        }
        switch (method)
        {
        case 2:
            mean = std::pow(mean, 1./data.size());
            break;
        case 3:
            mean = data.size() / mean;
            break;
        case 1:
        default:
            mean /= data.size();
            break;
        }
        // 误差平方倒数作权重
        // 递推式数学推导？减少计算次数
        for (double &sample_ : data) {
            inv_var += 1./((sample_ - mean) * (sample_ - mean));
        }
        sample = 0;
        for (double &sample_ : data) {
            sample += (1./((sample_-mean) * (sample_-mean))) / inv_var * sample_;
        }
    }
}

void Enemy::area_judge(const int &idx1, const int &idx2, int &main_id, int &sub_id) {
    if (armors[idx1].area_2d > armors[idx2].area_2d) {
        main_id = idx1;
        sub_id = idx2;
    } else {
        main_id = idx2;
        sub_id = idx1;
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
        int sub_armor_id = -1;
        int absent_armor_id = -1;
        bool ekf_init_flag = true; // 判断是否开始跟踪，初始化滤波器
        bool double_track_init_flag = false;
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
                if (armor.tracking_in_enemy) {
                    enemy.tracking_absent_flag = false;
                } else if (armor.sub_tracking_in_enemy) {
                    enemy.sub_tracking_absent_flag = false;
                }
            } else if (armor.status == Status::Absent && (armor.tracking_in_enemy || armor.sub_tracking_in_enemy)) {
                if (absent_armor_id == -1) {
                    absent_armor_id = i;
                } else {
                    RCLCPP_ERROR(get_logger(), "[update_enemy] Impossible more than 1 armors absent!");
                }
                if (armor.tracking_in_enemy) {
                    enemy.tracking_absent_flag = true;
                } else {
                    enemy.sub_tracking_absent_flag = true;
                }
            }
            if (armor.tracking_in_enemy || armor.sub_tracking_in_enemy) {
                ekf_init_flag = false;
            }
        }
        enemy.t_absent = recv_detection.time_stamp - enemy.alive_ts;

        // 更新装甲板数量
        enemy.armor_cnt = get_armor_cnt(static_cast<armor_type>(enemy.id % 9));
        if (enemy.armor_cnt != 3) {
            if (enemy.armor_cnt == 2) {
                enemy.balance_judge.update(1);
            } else {
                enemy.balance_judge.update(-1);
            }
            if (enemy.balance_judge.get() > 0) {
                enemy.armor_cnt = 2;
            } else {
                enemy.armor_cnt = 4;
            }
        }

        // 没有检测到装甲板，不进行更新
        if (enemy.status == Status::Absent) {
            continue;
        }
        // 更新dis2d
        enemy.min_dis_2d = INFINITY;
        for (TargetArmor &armor : enemy.armors) {
            enemy.min_dis_2d = std::min(enemy.min_dis_2d, armor.dis_2d);
        }

        double interal_r = 0.2;
        if (alive_indexs.size() == 1) {
            if (enemy.tracking_absent_flag && enemy.sub_tracking_absent_flag) {
                // 完全丢失，重置
                double_track_init_flag = true;
                tracking_armor_id = alive_indexs[0];
            } else if (!enemy.tracking_absent_flag && !enemy.sub_tracking_absent_flag) {
                tracking_armor_id = alive_indexs[0];
            } else if (enemy.sub_tracking_absent_flag || enemy.armors[alive_indexs[0]].tracking_in_enemy) {
                tracking_armor_id = alive_indexs[0];
            } else if (enemy.tracking_absent_flag || enemy.armors[alive_indexs[0]].sub_tracking_in_enemy) {
                sub_armor_id = alive_indexs[0];
            }
            enemy.double_track = false;
        } else if (alive_indexs.size() == 2) {
            if (absent_armor_id != -1) {
                TargetArmor &absent_armor = enemy.armors[absent_armor_id];
                double absent_armor_yaw = absent_armor.position_data.yaw;
                for (int &idx : alive_indexs) {
                    TargetArmor &armor = enemy.armors[idx];
                    double armor_yaw = armor.position_data.yaw;
                    if (fabs(armor_yaw - absent_armor_yaw) > (1 - interal_r) * M_PI && fabs(armor_yaw - absent_armor_yaw) < (1 + interal_r) * M_PI) {
                        if (absent_armor.tracking_in_enemy) {
                            tracking_armor_id = idx;
                            sub_armor_id = (alive_indexs[0] != idx) ? alive_indexs[0] : alive_indexs[1];
                        } else if (absent_armor.sub_tracking_in_enemy) {
                            sub_armor_id = idx;
                            tracking_armor_id = (alive_indexs[0] != idx) ? alive_indexs[0] : alive_indexs[1];
                        }
                        break;
                    }
                }
                if (tracking_armor_id == -1 && sub_armor_id == -1) {
                    RCLCPP_ERROR(get_logger(), "[update_enemy] no paired absent armor!");
                    // 从alive中挑选一个面积大的
                    enemy.area_judge(alive_indexs[0], alive_indexs[1], tracking_armor_id, sub_armor_id);
                    ekf_init_flag = true;
                }
            } else {
                // sub_tracking_in_enemy不一定一直有被设定，故||弱判断
                if (enemy.armors[alive_indexs[0]].tracking_in_enemy || enemy.armors[alive_indexs[1]].sub_tracking_in_enemy) {
                    tracking_armor_id = alive_indexs[0];
                    sub_armor_id = alive_indexs[1];
                } else if (enemy.armors[alive_indexs[0]].sub_tracking_in_enemy || enemy.armors[alive_indexs[1]].tracking_in_enemy) {
                    tracking_armor_id = alive_indexs[1];
                    sub_armor_id = alive_indexs[0];
                } else {
                    enemy.area_judge(alive_indexs[0], alive_indexs[1], tracking_armor_id, sub_armor_id);
                    double_track_init_flag = true;
                }
            }
            if (!enemy.double_track) {
                double_track_init_flag = true;
            }
            enemy.double_track = true;
        } else {
            RCLCPP_ERROR(get_logger(), "[update_enemy] impossible alive armor in one Enemy!");
        }
        RCLCPP_INFO(get_logger(), "[update_enemy] tracking_armor_id: %d", tracking_armor_id);
        if (enemy.enemy_ekf_init) {
            ekf_init_flag = false;
        }

        enemy.tracking_absent_flag = tracking_armor_id == -1;
        enemy.sub_tracking_absent_flag = sub_armor_id == -1;
        if (enemy.tracking_absent_flag && enemy.sub_tracking_absent_flag) {
            RCLCPP_ERROR(get_logger(), "[update_enemy] impossible tracking armor id!");
            continue;
        }
        
        // 设置tracking状态
        TargetArmor* tracking_armor = nullptr, *sub_armor = nullptr;
        if (!enemy.tracking_absent_flag) {
            tracking_armor = &enemy.armors[tracking_armor_id];
            tracking_armor->tracking_in_enemy = true;
        }
        if (!enemy.sub_tracking_absent_flag) {
            sub_armor = &enemy.armors[sub_armor_id];
            sub_armor->sub_tracking_in_enemy = true;
        }
        for (int i = 0; i < (int)enemy.armors.size(); ++i) {
            if (i != tracking_armor_id && !enemy.tracking_absent_flag) {
                enemy.armors[i].tracking_in_enemy = false;
            }
            if (i != sub_armor_id && !enemy.sub_tracking_absent_flag) {
                enemy.armors[i].sub_tracking_in_enemy = false;
            }
        }

        // 开始更新滤波
        // 计算enemy-yaw值
        double main_armor_yaw = tracking_armor_id != -1 ? tracking_armor->position_data.yaw : sub_armor->position_data.yaw + 2*M_PI/enemy.armor_cnt * ((enemy.last_yaw > enemy.last_yaw2) ? 1 : -1);
        double sub_armor_yaw = enemy.double_track ? sub_armor->position_data.yaw : main_armor_yaw + 2*M_PI/enemy.armor_cnt * ((enemy.last_yaw2 > enemy.last_yaw) ? 1 : -1);
        if (ekf_init_flag) {
            enemy.last_yaw = main_armor_yaw;
            enemy.last_yaw2 = sub_armor_yaw;
            enemy.yaw_round = 0;
            enemy.yaw2_round = 0;
            RCLCPP_WARN(get_logger(), "[update_enemy] armor_jump!");
        }

        // 处理过0
        if (main_armor_yaw - enemy.last_yaw < -M_PI * 1.5) {
            ++enemy.yaw_round;
        } else if (main_armor_yaw - enemy.last_yaw > M_PI * 1.5) {
            --enemy.yaw_round;
        }
        if (sub_armor_yaw - enemy.last_yaw2 < -M_PI * 1.5) {
            ++enemy.yaw2_round;
        } else if (sub_armor_yaw - enemy.last_yaw2 > M_PI * 1.5) {
            --enemy.yaw2_round;
        }
        double theta1 = main_armor_yaw + (enemy.yaw_round) * 2 * M_PI;
        double theta2 = sub_armor_yaw + (enemy.yaw2_round) * 2 * M_PI;
        enemy.last_yaw = main_armor_yaw;
        enemy.last_yaw2 = sub_armor_yaw;
        enemy.yaw = theta1;
        enemy.yaw2 = theta2;
        
        enemy_double_observer_EKF::Observe now_observe, now_observe2;
        enemy_double_observer_EKF::Observe2 now_double_observe;
        if (tracking_armor_id != -1) {
            now_observe.x = tracking_armor->getpos_xyz()[0];
            now_observe.y = tracking_armor->getpos_xyz()[1];
            now_observe.z = tracking_armor->getpos_xyz()[2];
            now_observe.yaw = theta1;
        }
        if (sub_armor_id != -1) {
            now_observe2.x = sub_armor->getpos_xyz()[0];
            now_observe2.y = sub_armor->getpos_xyz()[1];
            now_observe2.z = sub_armor->getpos_xyz()[2];
            now_observe2.yaw = theta2;
        }
        
        double ob_r1, ob_r2;
        if (enemy.double_track) {
            // Z = [x_1, y_1, z_1, r_1, theta_1, x_2, y_2, z_2, r_2, theta_2]
            double r1 = ((now_observe.x - now_observe2.x) * sin(theta2) - (now_observe.y - now_observe2.y) * cos(theta2)) / sin(theta2 - theta1);
            double r2 = ((now_observe2.x - now_observe.x) * sin(theta1) - (now_observe2.y - now_observe.y) * cos(theta1)) / sin(theta1 - theta2);
            ob_r1 = r1;
            ob_r2 = r2;
            enemy.observe_filter(enemy.r_data_set[0], r1, 3, (r1 < 0.24 && r1 > 0.12));
            enemy.observe_filter(enemy.r_data_set[1], r2, 3, (r2 < 0.24 && r2 > 0.12));
            now_double_observe.x = now_observe.x;
            now_double_observe.y = now_observe.y;
            now_double_observe.z = now_observe.z;
            now_double_observe.yaw = theta1;
            now_double_observe.x2 = now_observe2.x;
            now_double_observe.y2 = now_observe2.y;
            now_double_observe.z2 = now_observe2.z;
            now_double_observe.yaw2 = theta2;
            now_double_observe.r = r1;
            now_double_observe.r2 = r2;
        }
        
        if (fabs(enemy.ekf.state.vz) < 1e-2) {
            // z观测数据方差过大，适当处理
            if (tracking_armor_id != -1) {
                enemy.observe_filter(enemy.z_data_set[0], now_observe.z, 1, true);
            }
            if (sub_armor_id != -1) {
                enemy.observe_filter(enemy.z_data_set[1], now_observe2.z, 1, true);
            }
            if (enemy.double_track) {
                now_double_observe.z = now_observe.z;
                now_double_observe.z2 = now_observe2.z;
                enemy.dz = fabs(now_double_observe.z2 - now_double_observe.z);
                enemy.observe_filter(enemy.dz_data_set, enemy.dz, 1, true);
            }
        } else {
            enemy.z_data_set[0].clear();
            enemy.z_data_set[1].clear();
        }


        if (double_track_init_flag) {
            if (enemy.armor_cnt == 4) {
                if (enemy.double_track) {
                    if (fabs(now_double_observe.r2) < 0.28 && fabs(now_double_observe.r2) > 0.1) {
                        enemy.ekf.state.r2 = now_double_observe.r2;
                    }
                    if (fabs(now_double_observe.r) < 0.28 && fabs(now_double_observe.r) > 0.1) {
                        enemy.ekf.state.r = now_double_observe.r;
                    }
                    enemy.ekf.state.z2 = now_observe2.z;
                    double ob2_convince = 0.2; // 观测二置信度
                    enemy.ekf.state.x = (1-ob2_convince)*(now_observe.x - now_double_observe.r*cos(now_observe.yaw)) + ob2_convince*(now_observe2.x - now_double_observe.r2*cos(now_observe2.yaw));
                    enemy.ekf.state.y = (1-ob2_convince)*(now_observe.y - now_double_observe.r*sin(now_observe.yaw)) + ob2_convince*(now_observe2.y - now_double_observe.r2*sin(now_observe2.yaw));
                } else {
                    enemy.ekf.state.x = now_observe.x - enemy.ekf.state.r*cos(now_observe.yaw);
                    enemy.ekf.state.y = now_observe.y - enemy.ekf.state.r*sin(now_observe.yaw);
                }
            }
            if (tracking_armor_id != -1) {
                enemy.ekf.state.z = now_observe.z;
            }
            if (sub_armor_id != -1) {
                enemy.ekf.state.z2 = now_observe2.z;
            }
            enemy.z_data_set[0].clear();
            enemy.z_data_set[1].clear();
            enemy.ekf.state.vz = 0;
            enemy.ekf.state.yaw = theta1;
            enemy.ekf.state.yaw2 = theta2;
        }
        
        // // Raw Points Tracking Method
        // enemy_double_observer_EKF::Observe_pts now_observe_point, now_observe2_point;
        // enemy_double_observer_EKF::Observe_pts2 now_double_observe_point;
        // if (tracking_armor_id != -1) {
        //     now_observe_point.x1 = tracking_armor->position_data.img_pts[0].x;
        //     now_observe_point.y1 = tracking_armor->position_data.img_pts[0].y;
        //     now_observe_point.x2 = tracking_armor->position_data.img_pts[1].x;
        //     now_observe_point.y2 = tracking_armor->position_data.img_pts[1].y;
        //     now_observe_point.x3 = tracking_armor->position_data.img_pts[2].x;
        //     now_observe_point.y3 = tracking_armor->position_data.img_pts[2].y;
        //     now_observe_point.x4 = tracking_armor->position_data.img_pts[3].x;
        //     now_observe_point.y4 = tracking_armor->position_data.img_pts[3].y;
        // }
        // if (sub_armor_id != -1) {
        //     now_observe2_point.x1 = sub_armor->position_data.img_pts[0].x;
        //     now_observe2_point.y1 = sub_armor->position_data.img_pts[0].y;
        //     now_observe2_point.x2 = sub_armor->position_data.img_pts[1].x;
        //     now_observe2_point.y2 = sub_armor->position_data.img_pts[1].y;
        //     now_observe2_point.x3 = sub_armor->position_data.img_pts[2].x;
        //     now_observe2_point.y3 = sub_armor->position_data.img_pts[2].y;
        //     now_observe2_point.x4 = sub_armor->position_data.img_pts[3].x;
        //     now_observe2_point.y4 = sub_armor->position_data.img_pts[3].y;
        // }
        // if (enemy.double_track) {
        //     now_double_observe_point.x1a = tracking_armor->position_data.img_pts[0].x;
        //     now_double_observe_point.y1a = tracking_armor->position_data.img_pts[0].y;
        //     now_double_observe_point.x2a = tracking_armor->position_data.img_pts[1].x;
        //     now_double_observe_point.y2a = tracking_armor->position_data.img_pts[1].y;
        //     now_double_observe_point.x3a = tracking_armor->position_data.img_pts[2].x;
        //     now_double_observe_point.y3a = tracking_armor->position_data.img_pts[2].y;
        //     now_double_observe_point.x4a = tracking_armor->position_data.img_pts[3].x;
        //     now_double_observe_point.y4a = tracking_armor->position_data.img_pts[3].y;
        //     now_double_observe_point.x1b = sub_armor->position_data.img_pts[0].x;
        //     now_double_observe_point.y1b = sub_armor->position_data.img_pts[0].y;
        //     now_double_observe_point.x2b = sub_armor->position_data.img_pts[1].x;
        //     now_double_observe_point.y2b = sub_armor->position_data.img_pts[1].y;
        //     now_double_observe_point.x3b = sub_armor->position_data.img_pts[2].x;
        //     now_double_observe_point.y3b = sub_armor->position_data.img_pts[2].y;
        //     now_double_observe_point.x4b = sub_armor->position_data.img_pts[3].x;
        //     now_double_observe_point.y4b = sub_armor->position_data.img_pts[3].y;
        // }

        if (ekf_init_flag) {
            enemy.enemy_ekf_init = true;
            if (enemy.double_track) {
                enemy.ekf.reset2(now_double_observe);
            } else {
                if (tracking_armor_id != -1) {
                    enemy.ekf.reset(now_observe, true);
                } else {
                    enemy.ekf.reset(now_observe2, false);
                }
            }
            enemy.dz = 0;
            enemy.last_update_ekf_ts = enemy.alive_ts;
            RCLCPP_WARN(get_logger(), "armor_init! %ld", enemy.armors.size());
        } else {
            if (enemy.double_track) {
                // enemy.ekf.CKF_update2(enemy_double_observer_EKF::get_Z(now_double_observe_point), is_big_armor(static_cast<armor_type>(enemy.id % 9)), enemy.alive_ts - enemy.last_update_ekf_ts);              
                // enemy.ekf.CKF_update(enemy_double_observer_EKF::get_Z(now_observe_point), enemy_double_observer_EKF::get_Z(now_observe2_point), is_big_armor(static_cast<armor_type>(enemy.id % 9)), enemy.alive_ts - enemy.last_update_ekf_ts);
                enemy.ekf.CKF_update2(enemy_double_observer_EKF::get_Z(now_double_observe), enemy.alive_ts - enemy.last_update_ekf_ts);
                // enemy.ekf.update2(now_double_observe, enemy.alive_ts - enemy.last_update_ekf_ts);
            }
            else {
                // 状态保守更新
                if (tracking_armor_id != -1) {
                    enemy.ekf.state.vyaw2 = enemy.ekf.state.vyaw;
                    if (!double_track_init_flag) {
                        enemy.ekf.state.z2 = enemy.ekf.state.z + ((enemy.ekf.state.z2 > enemy.ekf.state.z) ? 1 : -1) * enemy.dz;
                    }
                    enemy.ekf.CKF_update(enemy_double_observer_EKF::get_Z(now_observe), enemy.alive_ts - enemy.last_update_ekf_ts, true);
                    // enemy.ekf.update(now_observe, enemy.alive_ts - enemy.last_update_ekf_ts);
                    // enemy.ekf.CKF_update(enemy_double_observer_EKF::get_Z(now_observe_point), is_big_armor(static_cast<armor_type>(enemy.id % 9)), true, enemy.alive_ts - enemy.last_update_ekf_ts);
                } else {
                    enemy.ekf.state.vyaw = enemy.ekf.state.vyaw2;
                    if (!double_track_init_flag) {
                        enemy.ekf.state.z = enemy.ekf.state.z2 + ((enemy.ekf.state.z > enemy.ekf.state.z2) ? 1 : -1) * enemy.dz;
                    }
                    enemy.ekf.CKF_update(enemy_double_observer_EKF::get_Z(now_observe2), enemy.alive_ts - enemy.last_update_ekf_ts, false);
                    // enemy.ekf.update(now_observe2, enemy.alive_ts - enemy.last_update_ekf_ts);
                    // enemy.ekf.CKF_update(enemy_double_observer_EKF::get_Z(now_observe2_point), is_big_armor(static_cast<armor_type>(enemy.id % 9)), false, enemy.alive_ts - enemy.last_update_ekf_ts);
                }
            }
            enemy.last_update_ekf_ts = enemy.alive_ts;
        }

        if (params.debug) {
            // 显示一下法向量
            if (tracking_armor_id != -1) {
                cv::Point2d show_pt_st = pc.pos2img(tracking_armor->getpos_xyz());
                cv::Point2d show_pt_end = pc.pos2img(tracking_armor->getpos_xyz() + tracking_armor->position_data.show_vec);
                cv::line(recv_detection.img, show_pt_st, show_pt_end, cv::Scalar(255, 0, 0), 2);
            }
            if (sub_armor_id != -1) {
                cv::Point2d show_pt_st = pc.pos2img(sub_armor->getpos_xyz());
                cv::Point2d show_pt_end = pc.pos2img(sub_armor->getpos_xyz() + sub_armor->position_data.show_vec);
                cv::line(recv_detection.img, show_pt_st, show_pt_end, cv::Scalar(255, 0, 0), 2);
            }

            // Foxglove Visualize
            // std_msgs::msg::Float64 x_msg, y_msg;
            // x_msg.data = enemy.ekf.state.x + enemy.ekf.state.r * cos(enemy.ekf.state.yaw);
            // y_msg.data = enemy.ekf.state.y + enemy.ekf.state.r * sin(enemy.ekf.state.yaw);
            // watch_data_pubs[0]->publish(x_msg);
            // watch_data_pubs[1]->publish(y_msg);
            // x_msg.data = enemy.ekf.state.x + enemy.ekf.state.r2 * cos(enemy.ekf.state.yaw2);
            // y_msg.data = enemy.ekf.state.y + enemy.ekf.state.r2 * sin(enemy.ekf.state.yaw2);
            // watch_data_pubs[2]->publish(x_msg);
            // watch_data_pubs[3]->publish(y_msg);
            // x_msg.data = now_observe_point.x1;
            // y_msg.data = now_observe_point.y1;
            // watch_data_pubs[4]->publish(x_msg);
            // watch_data_pubs[5]->publish(y_msg);
            // if (enemy.double_track) {
            //     x_msg.data = now_observe2_point.x1;
            //     y_msg.data = now_observe2_point.y1;
            //     watch_data_pubs[6]->publish(x_msg);
            //     watch_data_pubs[7]->publish(y_msg);
            // }

            
            // z
            // 双装甲板dz 单装甲板时锚定一个z为z +/- dz
            std_msgs::msg::Float64 z_msg;
            z_msg.data = enemy.ekf.state.z;
            watch_data_pubs[0]->publish(z_msg);
            z_msg.data = enemy.ekf.state.z2;
            watch_data_pubs[1]->publish(z_msg);
            if (tracking_armor_id != -1) {
                z_msg.data = now_observe.z;
                watch_data_pubs[2]->publish(z_msg);
            }
            if (sub_armor_id != -1) {
                z_msg.data = now_observe2.z;
                watch_data_pubs[3]->publish(z_msg);
            }
            z_msg.data = enemy.dz;
            watch_data_pubs[4]->publish(z_msg);
            z_msg.data = enemy.ekf.state.vz;
            watch_data_pubs[5]->publish(z_msg);
            
            // theta
            // std_msgs::msg::Float64 theta_msg;
            // theta_msg.data = enemy.ekf.state.yaw;
            // watch_data_pubs[0]->publish(theta_msg);
            // theta_msg.data = enemy.ekf.state.yaw2;
            // watch_data_pubs[1]->publish(theta_msg);
            // theta_msg.data = theta1;
            // watch_data_pubs[2]->publish(theta_msg);
            // theta_msg.data = theta2;
            // watch_data_pubs[3]->publish(theta_msg);
            // theta_msg.data = enemy.ekf.state.vyaw;
            // watch_data_pubs[4]->publish(theta_msg);
            // theta_msg.data = enemy.ekf.state.vyaw2;
            // watch_data_pubs[5]->publish(theta_msg);
            // theta_msg.data = enemy.yaw_round;
            // watch_data_pubs[6]->publish(theta_msg);
            // theta_msg.data = enemy.yaw2_round;
            // watch_data_pubs[7]->publish(theta_msg);

            // r
            // enemy.r = enemy.ekf.state.r;
            // enemy.r2 = enemy.ekf.state.r2;
            // if (fabs(enemy.last_ob_r2 - ob_r2) > fabs(enemy.last_ob_r2 - ob_r1) || fabs(enemy.last_ob_r - ob_r1) > fabs(enemy.last_ob_r - ob_r2)) {
            //     std::swap(ob_r1, ob_r2);
            // }
            // if (fabs(enemy.last_r2 - enemy.r2) > fabs(enemy.last_r2 - enemy.r) || fabs(enemy.last_r - enemy.r) > fabs(enemy.last_r - enemy.r2)) {
            //     std::swap(enemy.r, enemy.r2);
            // }
            // // ob_r1, ob_r2: raw measurement
            // // now_observe2.r/r2
            // enemy.last_ob_r = ob_r1;
            // enemy.last_ob_r2 = ob_r2;
            // enemy.last_r = enemy.r;
            // enemy.last_r2 = enemy.r2;
            // std_msgs::msg::Float64 r_msg;
            // r_msg.data = enemy.ekf.state.r;
            // // r_msg.data = enemy.r;
            // watch_data_pubs[0]->publish(r_msg);
            // r_msg.data = enemy.ekf.state.r2;
            // // r_msg.data = enemy.r2;
            // watch_data_pubs[1]->publish(r_msg);
            // if (enemy.double_track) {
            //     r_msg.data = now_observe2.r;
            //     // r_msg.data = ob_r1;
            //     watch_data_pubs[2]->publish(r_msg);
            //     // r_msg.data = ob_r2;
            //     r_msg.data = now_observe2.r2;
            //     watch_data_pubs[3]->publish(r_msg);
            // }

            show_enemies = cv::Mat(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
            cv::circle(show_enemies, cv::Point2d(320, 320), 4, cv::Scalar(211, 0, 148), 4);
            Enemy::enemy_positions pos = enemy.get_positions();
            cv::circle(show_enemies, cv::Point2d(320 - pos.center[1] * 50, 320 - pos.center[0] * 50), 2, cv::Scalar(127, 255, 170), 2);
            cv::circle(show_enemies, cv::Point2d(320 - pos.armors[0][1] * 50, 320 - pos.armors[0][0] * 50), 2, cv::Scalar(128, 0, 128), 2);
            for (int i = 1; i < enemy.armor_cnt; ++i) {
                cv::circle(show_enemies, cv::Point2d(320 - pos.armors[i][1] * 50, 320 - pos.armors[i][0] * 50), 2, cv::Scalar(128, 128, 128), 2);
            }
            // Enemy::enemy_positions pos_predict = enemy.predict_positions(response_delay);
            // 反投影预测点到图像
            cv::circle(recv_detection.img, pc.pos2img(pos.armors[0]), 3, cv::Scalar(211, 0, 148), 5);
            cv::circle(recv_detection.img, pc.pos2img(pos.center), 3, cv::Scalar(255, 0, 0), 5);
            for (int i = 1; i < enemy.armor_cnt; ++i) {
                cv::circle(recv_detection.img, pc.pos2img(pos.armors[i]), 3, cv::Scalar(255, 255, 255), 5);
                cv::line(recv_detection.img, pc.pos2img(Eigen::Vector3d(pos.center[0], pos.center[1], pos.armors[i][2])), pc.pos2img(pos.armors[i]),
                         cv::Scalar(127, 255, 170), 3);
            }
            // 画当前观测点、法向量、phase
            if (tracking_armor_id != -1) {
                cv::circle(show_enemies, cv::Point2d(320 - now_observe.y * 50, 320 - now_observe.x * 50), 3, cv::Scalar(211, 0, 148), 1);
                cv::line(show_enemies, cv::Point2d(320, 320),
                         cv::Point2d(320 - tracking_armor->position_data.normal_vec[1] * 50, 320 - tracking_armor->position_data.normal_vec[0] * 50),
                         cv::Scalar(255, 0, 0), 2);
                cv::putText(show_enemies, "Phase:" + std::to_string(tracking_armor->phase_in_enemy), cv::Point2d(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.5,
                            cv::Scalar(0, 255, 0));
            }
            if (sub_armor_id != -1) {
                cv::circle(show_enemies, cv::Point2d(320 - now_observe2.y * 50, 320 - now_observe2.x * 50), 3, cv::Scalar(211, 0, 148), 1);
                cv::line(show_enemies, cv::Point2d(320, 320),
                         cv::Point2d(320 - sub_armor->position_data.normal_vec[1] * 50, 320 - sub_armor->position_data.normal_vec[0] * 50),
                         cv::Scalar(255, 0, 0), 2);
                cv::putText(show_enemies, "Phase:" + std::to_string(sub_armor->phase_in_enemy), cv::Point2d(5, 10), cv::FONT_HERSHEY_COMPLEX, 0.5,
                            cv::Scalar(0, 255, 0));
            }

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

    }
}
