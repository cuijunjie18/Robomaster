#include <enemy_predictor/EnPredictor_utils.h>
#include <enemy_predictor/enemy_predictor.h>

using namespace enemy_predictor;

armor_EKF::Vy TargetArmor::getpos_xyz() const { return position_data.xyz; }
armor_EKF::Vy TargetArmor::getpos_pyd() const { return xyz2pyd(position_data.xyz); }

void TargetArmor::initpos_xyz(const Position_Calculator::pnp_result &new_pb, const double TS) {
    // 更新点坐标
    position_data = new_pb;
    // 重置滤波器
    Eigen::Matrix<double, 3, 1> new_pyd = xyz2pyd(new_pb.xyz);
    status = Alive;
    first_ts = alive_ts = TS;
    yaw_round = 0;
    last_yaw = new_pyd[1];
    kf.reset(new_pyd);
}
// 滤波器更新接口，内部使用pyd进行KF更新

void TargetArmor::updatepos_xyz(const Position_Calculator::pnp_result &new_pb, const double TS) {
    // 更新点坐标
    position_data = new_pb;

    Eigen::Matrix<double, 3, 1> new_pyd = xyz2pyd(new_pb.xyz);
    // 进行yaw区间过零处理
    if (new_pyd[1] - last_yaw < -M_PI * 1.5)
        yaw_round++;
    else if (new_pyd[1] - last_yaw > M_PI * 1.5)
        yaw_round--;
    last_yaw = new_pyd[1];
    new_pyd[1] += yaw_round * M_PI * 2;

    kf.update(new_pyd, TS - alive_ts);

    Position_Calculator::pnp_result new_position = new_pb;
    if (new_position.yaw - last_yaw_pose < -M_PI * 1.5) {  //
        yaw_round_pose++;
    } else if (new_position.yaw - last_yaw_pose > M_PI * 1.5) {  //
        yaw_round_pose--;
    }
    last_yaw_pose = new_position.yaw;
    new_position.yaw += yaw_round_pose * M_PI * 2;
    yaw_filter.update(new_position.yaw);

    // 更新时间戳和状态
    alive_ts = TS;
    status = Alive;
}

void TargetArmor::zero_crossing(double datum) {
    double yaw = kf.Xe[1];
    yaw += yaw_round * M_PI * 2;
    while (yaw - datum < -M_PI * 1.5) {
        yaw_round++;
        yaw += M_PI * 2;
    }
    while (yaw - datum > M_PI * 1.5) {
        yaw_round--;
        yaw -= M_PI * 2;
    }
    kf.Xe[1] = yaw;  // 利用协方差与均值无关的性质，对SCKF的点进行yaw轴上的平移
    return;
}

int EnemyPredictorNode::get_armor_cnt(armor_type type) {
    switch (type) {
        case armor_type::SENTRY:
        case armor_type::HERO:
        case armor_type::ENGINEER:
            return 4;
        case armor_type::OUTPOST:
            return 3;
        case armor_type::STANDARD_1:
        case armor_type::STANDARD_2:
        case armor_type::STANDARD_3: {  // 若为大装甲板步兵，则是一定是平衡
            return enemy_armor_type[type] ? 2 : 4;
        }
        default:
            return 4;
    }
}

bool EnemyPredictorNode::is_big_armor(armor_type type) { return enemy_armor_type[type]; }

void EnemyPredictorNode::update_armors() {
    static std::vector<match_edge> match_edges;    // 匹配边
    static std::vector<match_armor> match_armors;  // 参与匹配的装甲板
    static std::vector<TargetArmor> new_armors;    // 本次检测需要添加的新装甲板
    static std::vector<cv::Point2d> pts(4);        // 四点坐标暂存
    match_edges.clear();
    match_armors.clear();
    new_armors.clear();
    for (int j = 0; j < (int)enemies.size(); ++j) {
        for (int k = 0; k < (int)enemies[j].armors.size(); ++k) {
            enemies[j].armors[k].matched = false;                            // 清空匹配标志位
            enemies[j].armors[k].last_status = enemies[j].armors[k].status;  // 储存上一帧状态
            enemies[j].armors[k].status = Absent;                            // 所有装甲板初始化为暂时离线
        }
    }
    for (int i = 0; i < (int)recv_detection.res.size(); ++i) {
        Armor now_detect_armor = recv_detection.res[i];
        //     /9:color    %9:type
        int now_armor_id = now_detect_armor.color * 9 + now_detect_armor.type;
        // 同色过滤
        if (now_armor_id / 9 == params.rmcv_id / 9) continue;
        // 滤掉过小的装甲板
        for (int j = 0; j < 4; ++j) pts[j] = now_detect_armor.pts[j];
        double pts_S = get_area_armor(now_detect_armor.pts);
        if (pts_S < params.size_limit) continue;

        // 通过识别更新size信息
        enemy_armor_type[now_detect_armor.type] = now_detect_armor.size;
        bool isBigArmor = is_big_armor(static_cast<armor_type>(now_armor_id % 9));
        // 装甲板宽高比，若过于倾斜，则滤去装甲板
        // 必须使用外接矩形！！
        // fix flip training bug
        if (pts[3].x + pts[2].x < pts[0].x + pts[1].x) {
            std::swap(pts[0], pts[3]);
            std::swap(pts[1], pts[2]);
        }

        double aspect = (pts[3].x + pts[2].x - pts[0].x - pts[1].x) / (pts[1].y + pts[2].y - pts[0].y - pts[3].y);
        if ((isBigArmor && aspect <= params.aspect_limit_big) || (!isBigArmor && aspect <= params.aspect_limit_small)) continue;

        if (std::any_of(pts.begin(), pts.end(), [=](auto p) {
                return p.x < params.bound_limit || p.x > recv_detection.img.cols - params.bound_limit || p.y < params.bound_limit ||
                       p.y > recv_detection.img.rows - params.bound_limit;
            })) {
            continue;
        }

        Position_Calculator::pnp_result now_pos, now_pos_old;
        Eigen::Vector3d pyd_pos;
        if ((isBigArmor && aspect <= params.rm_pnp_aspect_limit_big) || (!isBigArmor && aspect <= params.rm_pnp_aspect_limit_small)) {
            now_pos = pc.pnp(pts, isBigArmor);
        } else {
            now_pos = pc.rm_pnp(pts, isBigArmor);
        }
        now_pos_old = pc.pnp(pts, isBigArmor);

        static double last_pos_yaw = 0, last_pos_yaw_old = 0;
        if (i == 0) {
            std_msgs::msg::Float64 now_pos_diff, now_pos_msg;
            now_pos_msg.data = now_pos.yaw / M_PI * 180;
            now_pos_diff.data = (now_pos.yaw - last_pos_yaw) / M_PI * 180;
            if (abs(now_pos_diff.data) > 70) {
                now_pos_diff.data = 0;
            }
            last_pos_yaw = now_pos.yaw;
            // watch_data_pubs[2]->publish(now_pos_diff);
            watch_data_pubs[2]->publish(now_pos_msg);
            std_msgs::msg::Float64 now_pos_diff_old, now_pos_old_msg;
            now_pos_old_msg.data = now_pos_old.yaw / M_PI * 180;
            now_pos_diff_old.data = (now_pos_old.yaw - last_pos_yaw_old) / M_PI * 180;
            if (abs(now_pos_diff_old.data) > 70) {
                now_pos_diff_old.data = 0;
            }
            last_pos_yaw_old = now_pos_old.yaw;
            // watch_data_pubs[3]->publish(now_pos_diff_old);
            watch_data_pubs[3]->publish(now_pos_old_msg);
        }
        double now_pitch = asin(now_pos.normal_vec[2]);
        RCLCPP_INFO(get_logger(), "now_pitch: %lf", now_pitch);
        if (now_pitch > params.top_pitch_thresh * M_PI / 360 && now_armor_id % 9 >= 6) {  // 编号为建筑并且pitch超过一定范围，判定为顶装甲
            RCLCPP_INFO(get_logger(), "top_armor~~~~~~~~~~~~");
            now_armor_id = now_detect_armor.color * 9 + armor_type::TOP;
        }

        // now_pos.xyz.cwiseProduct(pnp_compensate_k);
        // now_pos.xyz+=pnp_compensate_b;
        RCLCPP_INFO(get_logger(), "pnp:%lf,%lf,%lf [%lf]", now_pos.xyz[0], now_pos.xyz[1], now_pos.xyz[2], now_pos.xyz.norm());
        pyd_pos = xyz2pyd(now_pos.xyz);
        RCLCPP_INFO(get_logger(), "pnp_pyd:%lf,%lf,%lf", pyd_pos[0] * 180.0 / M_PI, pyd_pos[1] * 180.0 / M_PI, pyd_pos[2] * 180.0 / M_PI);
        if (params.debug) {
            cv::circle(recv_detection.img, pc.pos2img(now_pos.xyz), 3, cv::Scalar(0, 255, 255), 5);
        }

        // 远距离限制
        if (now_pos.xyz.norm() > params.sight_limit) continue;
        // 高度限制
        if (fabs(now_pos.xyz[2]) > params.high_limit) continue;

        match_armors.emplace_back(now_pos, now_armor_id, i, isBigArmor, false, now_detect_armor.rect);

        for (int j = 0; j < (int)enemies.size(); ++j) {
            for (int k = 0; k < (int)enemies[j].armors.size(); ++k) {
                armor_EKF::Vy last_xyz_pos = enemies[j].armors[k].getpos_xyz();
                double dis = calc_surface_dis_xyz(last_xyz_pos, now_pos.xyz);
                // logger.info("matching_dis:{}",dis);
                if (dis < params.interframe_dis_thresh) {                   // 帧间差角小于阈值
                    if (enemies[j].armors[k].id == now_armor_id) dis -= 1;  // 如果ID相同，则优先匹配
                    match_edges.emplace_back(j, k, match_armors.size() - 1, dis);
                }
            }
        }
    }
    std::sort(match_edges.begin(), match_edges.end());

    // logger.info("match_edges:{}",match_edges.size());
    // 贪心匹配
    for (int i = 0; i < (int)match_edges.size(); ++i) {
        match_edge now_match = match_edges[i];
        int eidx = now_match.last_enemy_idx;
        int aidx = now_match.last_sub_idx;
        int nidx = now_match.now_idx;
        // 把匹配和更新分开放两个循环就可以避免上一次moore投票修改导致下一次idx不合法的情况
        // 下次来改 TODO
        // 话说根本没必要把matched放到targetarmor里面吧
        if (aidx >= (int)enemies[eidx].armors.size()) {
            // this armor has been deleted by Moore's vote, just a workaround [TODO]
            continue;
        }
        if (enemies[eidx].armors[aidx].matched) continue;
        if (match_armors[nidx].matched) continue;
        enemies[eidx].armors[aidx].matched = 1;
        match_armors[nidx].matched = 1;
        // 摩尔投票修正
        if (match_armors[nidx].armor_id == enemies[eidx].armors[aidx].id) {
            enemies[eidx].armors[aidx].vote_cnt++;
            if (enemies[eidx].armors[aidx].vote_cnt > params.id_inertia) {
                enemies[eidx].armors[aidx].vote_cnt = params.id_inertia;
            }
        } else {
            enemies[eidx].armors[aidx].vote_cnt--;
        }
        // logger.warn("more: {}",enemies[eidx].armors[aidx].vote_cnt);

        auto &pts = recv_detection.res[match_armors[nidx].detection_idx].pts;
        if (enemies[eidx].armors[aidx].vote_cnt > 0) {  // 成功匹配，更新信息
            RCLCPP_INFO(get_logger(), "MATCHED");

            // 更新点坐标、滤波器以及装甲板外接矩形
            // 这里全用数组指标显得很呆
            if (enemies[eidx].is_high_spd_rotate) {
                double yaw_l = enemies[eidx].mono_inc.front().second;
                double yaw_r = enemies[eidx].mono_dec.front().second;
                double yaw_middle = angle_middle(yaw_l, yaw_r);
                double yaw_las = enemies[eidx].armors[aidx].getpos_pyd()[1];
                double yaw_now = atan2(match_armors[nidx].position.xyz[1], match_armors[nidx].position.xyz[0]);
            }
            enemies[eidx].armors[aidx].updatepos_xyz(match_armors[nidx].position, recv_detection.time_stamp);
            enemies[eidx].armors[aidx].bounding_box = match_armors[nidx].bbox;
            enemies[eidx].common_yaw_spd.update(enemies[eidx].armors[aidx].get_yaw_spd());
            enemies[eidx].armors[aidx].dis_2d = get_dis2d(params.collimation, std::accumulate(pts + 1, pts + 4, pts[0]) / 4);  // 装甲四点到准星距离
            enemies[eidx].armors[aidx].area_2d = get_area_armor(pts);
        } else {
            // 处理摩尔投票计数归零的情况
            RCLCPP_WARN(get_logger(), "Moore's vote 0!");
            TargetArmor tmp = enemies[eidx].armors[aidx];
            enemies[eidx].armors.erase(enemies[eidx].armors.begin() + aidx);  // 移除原armor
            tmp.id = match_armors[nidx].armor_id;
            tmp.vote_cnt = 1;
            tmp.dis_2d = get_dis2d(params.collimation, std::accumulate(pts + 1, pts + 4, pts[0]) / 4);  // 装甲四点到准星距离
            tmp.area_2d = get_area_armor(pts);
            tmp.updatepos_xyz(match_armors[nidx].position, recv_detection.time_stamp);
            tmp.bounding_box = match_armors[nidx].bbox;
            tmp.tracking_in_enemy = false;

            new_armors.push_back(tmp);  // 摩尔投票归零， 也需要被视为新出现的装甲板进行处理?
        }
    }
    if (new_armors.size() > 0) {
        RCLCPP_WARN(get_logger(), "new armors from Moore: %ld", new_armors.size());
    }

    // 处理没有被匹配的新识别到的装甲板
    for (int i = 0; i < (int)match_armors.size(); ++i) {
        if (match_armors[i].matched) continue;

        TargetArmor tmp;

        auto &pts = recv_detection.res[match_armors[i].detection_idx].pts;
        tmp.dis_2d = get_dis2d(params.collimation, std::accumulate(pts + 1, pts + 4, pts[0]) / 4);  // 装甲四点到准星距离
        tmp.area_2d = get_area_armor(pts);

        tmp.id = match_armors[i].armor_id;
        tmp.vote_cnt = 1;
        tmp.initpos_xyz(match_armors[i].position, recv_detection.time_stamp);
        tmp.bounding_box = match_armors[i].bbox;
        tmp.tracking_in_enemy = false;
        new_armors.push_back(tmp);
    }
    if (new_armors.size() > 0) {
        RCLCPP_WARN(get_logger(), "new armors: %ld", new_armors.size());
    }
    // 添加新的识别装甲板
    for (int i = 0; i < (int)new_armors.size(); ++i) {
        if (new_armors[i].id / 9 == 2) {  // 过滤白色
            continue;
        }
        if (params.debug) {
            cv::circle(recv_detection.img, pc.pos2img(new_armors[i].getpos_xyz()), 27, cv::Scalar(255, 255, 255), -1);
        }
        bool enemy_exists = false;
        for (int j = 0; j < (int)enemies.size(); ++j) {
            if (enemies[j].id != new_armors[i].id) continue;
            enemy_exists = true;
            if (!enemies[j].mono_inc.empty()) {
                new_armors[i].zero_crossing(enemies[j].mono_inc.back().second);
            }
            enemies[j].add_armor(new_armors[i]);
            break;
        }
        if (!enemy_exists) {
            Enemy new_enemy(this);
            new_enemy.id = new_armors[i].id;
            new_enemy.enemy_ekf_init = false;
            new_enemy.following = false;
            new_enemy.armor_cnt = get_armor_cnt(static_cast<armor_type>(new_enemy.id % 9));
            // new_enemy.armor_cnt = 2;
            new_enemy.add_armor(new_armors[i]);
            // new_enemy.init_enemy_observer(new_armors[i], detections.time_stamp);
            enemies.push_back(new_enemy);
            RCLCPP_WARN(get_logger(), "new enemy: id: %d", new_enemy.id);
        }
    }
    // 处理过时的装甲板和敌对目标
    for (int i = 0; i < (int)enemies.size(); ++i) {
        for (auto it = enemies[i].armors.begin(); it != enemies[i].armors.end();) {
            // 删除Absent超过reset_time的装甲板
            if (it->alive_ts + params.reset_time < recv_detection.time_stamp) {
                it = enemies[i].armors.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto it = enemies.begin(); it != enemies.end();) {
        if (it->armors.empty()) {
            it = enemies.erase(it);
        } else {
            ++it;
        }
    }
    if (params.debug) {
        for (int i = 0; i < (int)enemies.size(); ++i) {
            for (int j = 0; j < (int)enemies[i].armors.size(); ++j) {
                TargetArmor &now_armor = enemies[i].armors[j];
                // cv::circle(recv_detection.img, pc.pos2img(pyd2xyz(now_armor.kf.predict(params.response_delay))), 3, cv::Scalar(0, 255, 0), 5);
                // cv::Point2d img_pos = pc.pos2img(pyd2xyz(now_armor.kf.predict(0)));
                // logger.sinfo("img pos: {}, {}", img_pos.x, img_pos.y);
                // Eigen::Vector3d pos = now_armor.kf.predict(0);
                // logger.sinfo("PYD: {}, {}, {}", pos[0], pos[1], pos[2]);
                // logger.warn("PYD: {}, {}, {}", pos[0], pos[1], pos[2]);
                // Eigen::Vector3d pos_xyz = pyd2xyz(now_armor.kf.predict(0));
                // logger.sinfo("XYZ: {}, {}, {}", pos_xyz[0], pos_xyz[1], pos_xyz[2]);
            }
        }
    }
}