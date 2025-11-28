// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "util/kmeans.h"
#include "util/threading.h"

namespace sensemap {

std::unique_ptr<ThreadPool> thread_pool;

void KMeans::MovetoCenter(){
    Eigen::Vector3d mean_coord(0, 0, 0);
    for (const auto& pnt : mv_pntcloud){
        mean_coord = mean_coord + pnt.location;
    }

    mean_coord = mean_coord / static_cast<double>(mv_pntcloud.size());
    
    for(auto&  pnt : mv_pntcloud){
        pnt.location = pnt.location - mean_coord;
    }
}

void KMeans::SetRandSeed(int seed){
    srand_seed = seed;
}

double KMeans::GetClosestDist(Tuple& point, std::vector<Tuple>& centers) {
    double min_dist = __DBL_MAX__;
    for (auto center : centers) {
        double dist = DistBetweenPoints(point, center);
        if (min_dist - dist > 0.0000000001) {
            min_dist = dist;
        }
    }
    return min_dist;
}

bool KMeans::PlusInit() {
    mv_center.clear();
    if (srand_seed < 0){
        srand(unsigned(12345));
    } else {
        srand(unsigned(srand_seed));
    }
    int size = mv_pntcloud.size();
    // int seed = random() % (size + 1);
    int seed = random() % (size);
    Tuple begin_center;
    begin_center.location[0] = mv_pntcloud[seed].location[0];
    begin_center.location[1] = mv_pntcloud[seed].location[1];
    begin_center.location[2] = mv_pntcloud[seed].location[2];
    mv_center.emplace_back(begin_center);
    std::vector<double> dist(size, 0.0);
    for (size_t i = 1; i < m_k; i++) {
        double tot = 0.0;
        for (size_t j = 0; j < size; j++) {
            dist[j] = GetClosestDist(mv_pntcloud[j], mv_center);
            tot += dist[j];
        }
        tot *= random() / double(RAND_MAX);
        for (size_t j = 0; j < size; j++) {
            tot -= dist[j];
            if (tot > 0.0000000001) {
                continue;
            }
            Tuple next_center;
            next_center.location[0] = mv_pntcloud[j].location[0];
            next_center.location[1] = mv_pntcloud[j].location[1];
            next_center.location[2] = mv_pntcloud[j].location[2];
            mv_center.emplace_back(next_center);
            break;
        }
    }
    return true;
}

bool KMeans::PlusCluster() {
    PlusInit();
    std::vector<Tuple> v_center(mv_center.size());
    size_t pntCount = mv_pntcloud.size();

    int iter_num = 0;

    do {
        for (size_t i = 0; i < pntCount; i++) {
            double min_dist = __DBL_MAX__;
            int pnt_grp = 0;
            for (size_t j = 0; j < m_k; j++) {
                double dist = DistBetweenPoints(mv_pntcloud[i], mv_center[j]);
                // std::cout << "dist: " << dist << std::endl;
                if (min_dist - dist > 0.00000000001) {
                    min_dist = dist;
                    pnt_grp = j;
                }
            }
            m_grp_pntcloud[pnt_grp].emplace_back(mv_pntcloud[i]);
        }

        
        // std::cout << " *** iter " << iter_num + 1 << std::endl;
	    for (size_t i = 0; i < mv_center.size(); i++) {
            v_center[i] = mv_center[i];
            // std::cout << i << " m_grp_pntcloud: " << m_grp_pntcloud[i].size() << std::endl;
            // std::cout << i << " mv_center: " << v_center[i].location << std::endl;
        }

        UpdateGroupCenter();
        if (!ExistCenterShift(v_center, mv_center) || iter_num >= max_iter) {
            std::cout << "iter_num: " << iter_num << std::endl;
            break;
        }
        for (size_t i = 0; i < m_k; ++i) {
            m_grp_pntcloud[i].clear();
            m_grp_pntcloud[i].shrink_to_fit();
        }
        iter_num++;
    } while (true);

    return true;
}

bool KMeans::InitKCenter() {
    mv_center.resize(m_k);
    int size = mv_pntcloud.size();
    if (srand_seed < 0){
        srand(unsigned(12345));
    } else {
        srand(unsigned(srand_seed));
    }
    for (size_t i = 0; i < m_k; i++) {
        int seed = random() % (size + 1);
        mv_center[i].location[0] = mv_pntcloud[seed].location[0];
        mv_center[i].location[1] = mv_pntcloud[seed].location[1];
        mv_center[i].location[2] = mv_pntcloud[seed].location[2];
    }
    return true;
}

bool KMeans::Cluster() {
    InitKCenter();
    std::vector<Tuple> v_center(mv_center.size());
    size_t pntCount = mv_pntcloud.size();

    int iter_num = 0;

    do {
        for (size_t i = 0; i < pntCount; i++) {
            double min_dist = __DBL_MAX__;
            int pnt_grp = 0;
            for (size_t j = 0; j < m_k; j++) {
                double dist = DistBetweenPoints(mv_pntcloud[i], mv_center[j]);
                // std::cout << "dist: " << dist << std::endl;
                if (min_dist - dist > 0.000001) {
                    min_dist = dist;
                    pnt_grp = j;
                }
            }
            m_grp_pntcloud[pnt_grp].emplace_back(mv_pntcloud[i]);
        }

        // std::cout << " *** " << std::endl;
        for (size_t i = 0; i < mv_center.size(); i++) {
            v_center[i] = mv_center[i];
            // std::cout << "m_grp_pntcloud: " << m_grp_pntcloud[i].size() << std::endl;
            // std::cout << "mv_center: " << v_center[i].location << std::endl;
        }

        UpdateGroupCenter();
        if (!ExistCenterShift(v_center, mv_center) || iter_num > max_iter) {
            // std::cout << "iter_num: " << iter_num << std::endl;
            break;
        }
        for (size_t i = 0; i < m_k; ++i) {
            m_grp_pntcloud[i].clear();
            m_grp_pntcloud[i].shrink_to_fit();
        }
        iter_num++;
    } while (true);

    return true;
}

double KMeans::DistBetweenPoints(const Tuple& p1, const Tuple& p2) {
    double dist = (p1.location - p2.location).norm();
    return dist;
}

bool KMeans::UpdateGroupCenter() {
    for (size_t i = 0; i < m_k; i++) {
        float x = 0, y = 0, z = 0;
        size_t pnt_num_in_grp = m_grp_pntcloud[i].size();

        if (pnt_num_in_grp > 0) {
            for (size_t j = 0; j < pnt_num_in_grp; j++) {
                x += m_grp_pntcloud[i][j].location[0];
                y += m_grp_pntcloud[i][j].location[1];
                z += m_grp_pntcloud[i][j].location[2];
            }
            x /= pnt_num_in_grp;
            y /= pnt_num_in_grp;
            z /= pnt_num_in_grp;

            mv_center[i].location[0] = x;
            mv_center[i].location[1] = y;
            mv_center[i].location[2] = z;
        } else {
            PlusInit();
            break;
        }
    }
    return true;
}

bool KMeans::ComputeGroupCenter() {
    for (size_t i = 0; i < m_k; i++) {
        float x = 0, y = 0, z = 0;
        size_t pnt_num_in_grp = m_grp_pntcloud[i].size();

        if (pnt_num_in_grp > 0) {
            for (size_t j = 0; j < pnt_num_in_grp; j++) {
                x += m_grp_pntcloud[i][j].location[0];
                y += m_grp_pntcloud[i][j].location[1];
                z += m_grp_pntcloud[i][j].location[2];
            }
            x /= pnt_num_in_grp;
            y /= pnt_num_in_grp;
            z /= pnt_num_in_grp;

            mv_center[i].location[0] = x;
            mv_center[i].location[1] = y;
            mv_center[i].location[2] = z;
        }
    }
    return true;
}

bool KMeans::ExistCenterShift(std::vector<Tuple>& prev_center, std::vector<Tuple>& cur_center) {
    for (size_t i = 0; i < m_k; i++) {
        double dist = DistBetweenPoints(prev_center[i], cur_center[i]);
        if (dist > DIST_NEAR_ZERO) {
            return true;
        }
    }

    return false;
}

bool KMeans::SameSizeCluster(){
    if (m_k * fixed_size < mv_pntcloud.size()){
        std::cout << "m_k * fix_size < mv_pntcloud.size()" << std::endl;
        return false;
    }

    if ( m_k > 1 && (m_k - 1) * fixed_size > mv_pntcloud.size()){
        std::cout << "(m_k-1) * fix_size > mv_pntcloud.size()" << std::endl;
        return false;
    }

    std::vector<VecPoint_t> m_grp_same_size;
    VecPoint_t mv_same_size_center;

    // std::vector<VecPoint_t> samesize_grp_pntcloud;
    // std::vector<Tuple> samesize_center;

    MovetoCenter();

    int k = m_k;
    while(m_k > 0){
        if (m_k > 1){
            //Cluster();
	        PlusCluster();
    	    ComputeGroupCenter();
            
            mv_pntcloud.clear();
            
            // Calculate numMem
            Eigen::Vector3d mean_location = Eigen::Vector3d::Zero();
            int sum_points = 0;
            float x = 0, y = 0, z = 0;
            for (int center_idx = 0; center_idx < m_k; center_idx++){
                x += mv_center[center_idx].location[0] * m_grp_pntcloud[center_idx].size();
                y += mv_center[center_idx].location[1] * m_grp_pntcloud[center_idx].size();
                z += mv_center[center_idx].location[2] * m_grp_pntcloud[center_idx].size();
                sum_points += m_grp_pntcloud[center_idx].size();
                std::cout << m_k << " iter " << center_idx << " mv_center: "
                << "numpoints:" << m_grp_pntcloud[center_idx].size() 
                            << " xyz:" << mv_center[center_idx].location[0] 
                            << "," << mv_center[center_idx].location[1] 
                            << "," << mv_center[center_idx].location[2] << std::endl;
            }
            x /= sum_points;
            y /= sum_points;
            z /= sum_points;
            mean_location[0] = x;
            mean_location[1] = y;
            mean_location[2] = z;
            std::cout << "mean location is : " << mean_location[0] 
                << "," << mean_location[1] << "," << mean_location[2] 
                << " size: " << sum_points << std::endl;
            for (int center_idx = 0; center_idx < m_k; center_idx++){
                x = mv_center[center_idx].location[0] - mean_location[0];
                y = mv_center[center_idx].location[1] - mean_location[1];
                z = mv_center[center_idx].location[2] - mean_location[2];
                std::cout << m_k << " iter " << center_idx << " xyz / dist: " 
                            << x << " " << y << " " << z << " / "
                            << x*x + y*y + z*z << std::endl;
            }

            // find the furthest center
            int max_dist_idx = 0;
            double max_dist = 0;
            for (int center_idx = 0; center_idx < m_k; center_idx++){
                double dist_center = 
                    (mv_center[center_idx].location - mean_location).norm();
                if (dist_center > max_dist){
                    max_dist = dist_center;
                    max_dist_idx = center_idx;
                }
            }
            std::cout << "max_dist_idx: " << max_dist_idx << std::endl;

            // whether we recurit or discard points
            VecPoint_t grp_same_size;
            if (m_grp_pntcloud[max_dist_idx].size() > fixed_size){
                int num_discard = m_grp_pntcloud[max_dist_idx].size() - fixed_size;

                int min_idx;
                double min_dist = std::numeric_limits<double>::max();
                for (int center_idx = 0; center_idx < m_k; center_idx++){
                    if (center_idx == max_dist_idx){
                        continue;
                    }
                    double dist = (mv_center[center_idx].location - 
                        mv_center[max_dist_idx].location).norm();
                    if (dist < min_dist){
                        min_dist = dist;
                        min_idx = center_idx;
                    }
                }

                std::vector<std::pair<int, double>> grp_dist;
                grp_dist.resize(m_grp_pntcloud[max_dist_idx].size());
                for (int pnt_id = 0; pnt_id < m_grp_pntcloud[max_dist_idx].size(); pnt_id++){
                    double dist = DistBetweenPoints(mv_center[min_idx],
                                  m_grp_pntcloud[max_dist_idx].at(pnt_id));
                    grp_dist[pnt_id] = std::pair<int, double>(pnt_id, dist);
                }
                std::sort(grp_dist.begin(),grp_dist.end(), 
                          [](const std::pair<int, double> & dist1,
                          const std::pair<int, double> & dist2) {
                          return dist1.second > dist2.second;});

                // remove the point from center_id
                for (int id = 0; id < fixed_size; id++){
                    grp_same_size.push_back(
                        m_grp_pntcloud[max_dist_idx].at(grp_dist[id].first));
                }

                // update mv_pntcloud
                for (int id = fixed_size; 
                     id < m_grp_pntcloud[max_dist_idx].size(); id++){
                    mv_pntcloud.push_back(m_grp_pntcloud[max_dist_idx]
                                          .at(grp_dist[id].first));
                }
                for (int center_idx = 0; center_idx < m_k; center_idx++){
                    if (center_idx == max_dist_idx){
                        continue;
                    }
                    mv_pntcloud.insert(mv_pntcloud.end(), 
                                    m_grp_pntcloud[center_idx].begin(), 
                                    m_grp_pntcloud[center_idx].end());
                }                
            }else if (m_grp_pntcloud[max_dist_idx].size() == fixed_size){
                grp_same_size = m_grp_pntcloud[max_dist_idx];
                for (int center_idx = 0; center_idx < m_k; center_idx++){
                    if (center_idx == max_dist_idx){
                        continue;
                    }
                    mv_pntcloud.insert(mv_pntcloud.end(), 
                                       m_grp_pntcloud[center_idx].begin(), 
                                       m_grp_pntcloud[center_idx].end());
                }      
            }else if (m_grp_pntcloud[max_dist_idx].size() < fixed_size){
                struct GrpIdx{
                    int center_id;
                    int pnt_id;
                };
                std::vector<std::pair<GrpIdx, double>> grp_dist;
                
                for (int center_idx = 0; center_idx < m_k; center_idx++){
                    if (center_idx == max_dist_idx){
                        continue;
                    }
                    for (int pnt_idx = 0; pnt_idx < m_grp_pntcloud[center_idx].size();
                         pnt_idx++){
                        double dist = DistBetweenPoints(mv_center[max_dist_idx],
                                    m_grp_pntcloud[center_idx].at(pnt_idx));
                        GrpIdx grp_id{center_idx, pnt_idx};
                        grp_dist.push_back(std::pair<GrpIdx, double>(grp_id, dist));
                    }
                }
                std::sort(grp_dist.begin(),grp_dist.end(), 
                          [](const std::pair<GrpIdx, double> & dist1,
                          const std::pair<GrpIdx, double> & dist2) {
                          return dist1.second < dist2.second;});
                
                grp_same_size = m_grp_pntcloud[max_dist_idx];
                // std::cout << "grp_same_size" << grp_same_size.size() << std::endl;
                for (int i = 0; i < fixed_size - m_grp_pntcloud[max_dist_idx].size(); i++){
                    int center_id = grp_dist.at(i).first.center_id;
                    int pnt_id = grp_dist.at(i).first.pnt_id;
                    grp_same_size.push_back(m_grp_pntcloud[center_id].at(pnt_id));
                    m_grp_pntcloud[center_id].at(pnt_id).dist = -1.0;
                }
                // std::cout << "grp_same_size" << grp_same_size.size() << std::endl;

                for (int center_idx = 0; center_idx < m_k; center_idx++){
                    if (center_idx == max_dist_idx){
                        continue;
                    }
                    mv_pntcloud.insert(mv_pntcloud.end(), 
                                       m_grp_pntcloud[center_idx].begin(), 
                                       m_grp_pntcloud[center_idx].end());
                }
                // std::cout << "mv_pntcloud size: " << mv_pntcloud.size() << std::endl;
                for(auto it = mv_pntcloud.begin(); it != mv_pntcloud.end(); it++){
                    if(it->dist < -0.01){
                        it = mv_pntcloud.erase(it);
                        it--;
                        if(it == mv_pntcloud.end()) break;
                    }
                }
                // std::cout << "mv_pntcloud size: " << mv_pntcloud.size() << std::endl;
            }

            m_grp_same_size.push_back(grp_same_size);
            mv_same_size_center.push_back(mv_center[max_dist_idx]);
            m_grp_pntcloud.clear();
            mv_center.clear();

            // std::cout << "mv_pntcloud: " << mv_pntcloud.size() << std::endl;
            // std::cout << "grp_same_size" << grp_same_size.size() << std::endl;
            m_k--;
            m_grp_pntcloud.resize(m_k);
        } else if (m_k == 1){
            Cluster();
            m_grp_same_size.push_back(m_grp_pntcloud.at(0));
            mv_same_size_center.push_back(mv_center.at(0));
            m_grp_pntcloud.clear();
            mv_center.clear();
            m_k--;
        }
    }

    m_grp_pntcloud.swap(m_grp_same_size);
    mv_center.swap(mv_same_size_center);
    m_k = k;

    ComputeGroupCenter();
    std::cout << "m_grp_pntcloud size: ";
    for (int i = 0; i < m_grp_pntcloud.size(); i++){
        std::cout << m_grp_pntcloud.at(i).size() << " ";;
    }
    std::cout << std::endl;

    return true;
}

bool cmp_dist(Tuple& a, Tuple& b) { return a.dist < b.dist; }

void KMeans::FindNeighborsAndCommonPoints(
    std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
    std::vector<std::vector<int>>& neighbors) {
    neighbors_points.resize(m_k);
    neighbors.resize(m_k);
    // find common images.
    for (size_t i = 0; i < m_k; i++) {
        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            int neighbor;
            double min_dist = __DBL_MAX__;
            for (size_t k = 0; k < m_k; k++) {
                if (i == k) {
                    continue;
                }
                double dist = DistBetweenPoints(m_grp_pntcloud[i][j], mv_center[k]);
                if (min_dist - dist > 0.000001) {
                    min_dist = dist;
                    neighbor = k;
                }
            }
            m_grp_pntcloud[i][j].dist = min_dist;
            neighbors_points[i][neighbor].emplace_back(m_grp_pntcloud[i][j]);
        }
    }
    // find neighbors.
    for (size_t i = 0; i < m_k; i++) {
        // neighbors[i].reserve(neighbors_points[i].size());
        std::cout << i << " neighbors:";
        for (auto& neighbor_data : neighbors_points[i]) {
            sort(neighbor_data.second.begin(), neighbor_data.second.end(), cmp_dist);
            neighbors[i].emplace_back(neighbor_data.first);
            std::cout << " " << neighbor_data.first;
        }
        std::cout << std::endl;
    }
}

void KMeans::FindNeighborsAndCommonPoints_EdgeNearest(
    std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
    std::vector<std::vector<int>>& neighbors) {
    neighbors_points.resize(m_k);
    neighbors.resize(m_k);
    // find common images.
    for (size_t i = 0; i < m_k; i++) {
        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            int neighbor;
            double min_dist = __DBL_MAX__;
            for (size_t k = 0; k < m_k; k++) {
                if (i == k) {
                    continue;
                }
                double dist = DistBetweenPoints(m_grp_pntcloud[i][j], mv_center[k]);
                if (min_dist - dist > 0.000001) {
                    min_dist = dist;
                    neighbor = k;
                }
            }
            m_grp_pntcloud[i][j].dist = min_dist;
            neighbors_points[i][neighbor].emplace_back(m_grp_pntcloud[i][j]);
        }
    }
    // update min dist and find neighbors.
    for (size_t i = 0; i < m_k; i++) {
        // std::cout  << i << " neighbors:";
        for (auto& neighbor_data : neighbors_points[i]) {
            std::cout  << " " << neighbor_data.first;
            for (size_t j = 0; j < neighbor_data.second.size(); j++) {
                // std::cout  << "j: " << j <<std::endl;
                if (neighbor_data.second[j].name.empty()) {
                    std::cout  << "null neighbor_data.second[j]: " << j <<std::endl;
                    std::cout  << "size: " << neighbor_data.second.size() <<std::endl;
                    exit(1);
                }
                double min_dist = __DBL_MAX__;
                for (size_t k = 0; k < m_grp_pntcloud[neighbor_data.first].size(); k++) {
                    if (m_grp_pntcloud[neighbor_data.first][k].name.empty()) {
                        std::cout  << "neighbor_data.first: " << neighbor_data.first <<std::endl;
                        std::cout  << "size: " << m_grp_pntcloud[neighbor_data.first].size() <<std::endl;
                        std::cout  << "null k: " << k <<std::endl;
                        exit(1);
                    }
                    double dist = DistBetweenPoints(m_grp_pntcloud[neighbor_data.first][k],
                                                           neighbor_data.second[j]);
                    // if (0.1 - dist > 0.001){
                    //     std::cout  << "min_dist: " << dist <<std::endl;
                    //     std::cout  << "neighbor_data.second[j]: " << neighbor_data.second[j].name
                    //     << " " << neighbor_data.second[j].location <<std::endl;
                    //     std::cout  << "m_grp_pntcloud[neighbor_data.first][k]: "
                    //     << m_grp_pntcloud[neighbor_data.first][k].name
                    //     << " " << m_grp_pntcloud[neighbor_data.first][k].location <<std::endl;
                    //     exit(1);
                    // }

                    if (min_dist - dist > 0.000001) {
                        min_dist = dist;
                    }
                }
                neighbor_data.second[j].dist = min_dist;
            }
            neighbors[i].emplace_back(neighbor_data.first);
            // bubble sort
            for (size_t m = 0; m < neighbor_data.second.size()-1; m++) {
                for (size_t n = 0; n < neighbor_data.second.size()-1-m; n++) {
                    if (neighbor_data.second[n].dist - neighbor_data.second[n+1].dist > 0.001) {
                        auto temp = neighbor_data.second[n+1];
                        neighbor_data.second[n+1] = neighbor_data.second[n];
                        neighbor_data.second[n] = temp;
                    }
                }
            }
            // std::cout  << "(" << neighbor_data.second.size();
            // sort(neighbor_data.second.begin(), neighbor_data.second.end(), cmp_dist);
            // std::cout  << ")";
        }
        std::cout  <<std::endl;
    }
}

void KMeans::FindNeighborsAndCommonPoints_AllPoints(
    std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
    std::vector<std::vector<int>>& neighbors) {
    neighbors_points.resize(m_k);
    neighbors.resize(m_k);
    // find common images.
    for (size_t i = 0; i < m_k; i++) {
        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            int neighbor;
            double min_dist = __DBL_MAX__;
            for (size_t k = 0; k < m_k; k++) {
                if (i == k) {
                    continue;
                }
                for (size_t m = 0; m < m_grp_pntcloud[k].size(); m++) {
                    double dist = DistBetweenPoints(m_grp_pntcloud[i][j], m_grp_pntcloud[k][m]);
                    if (min_dist - dist > 0.000001) {
                        min_dist = dist;
                        neighbor = k;
                    }
                }
            }
            m_grp_pntcloud[i][j].dist = min_dist;
            neighbors_points[i][neighbor].emplace_back(m_grp_pntcloud[i][j]);
        }
    }
    std::cout  << "finish neighbors_points." <<std::endl;
    // find neighbors.
    for (size_t i = 0; i < m_k; i++) {
        // neighbors[i].reserve(neighbors_points[i].size());
        std::cout  << i << " neighbors:";
        for (auto& neighbor_data : neighbors_points[i]) {
            std::cout  << " " << neighbor_data.first;
            neighbor_data.second.shrink_to_fit();
            if (neighbor_data.second.empty()) {
                std::cout  << "neighbor_data.second.empty()" <<std::endl;
                exit(1);
            }
            // bubble sort
            for (size_t m = 0; m < neighbor_data.second.size()-1; m++) {
                for (size_t n = 0; n < neighbor_data.second.size()-1-m; n++) {
                    if (neighbor_data.second[n].dist - neighbor_data.second[n+1].dist > 0.001) {
                        auto temp = neighbor_data.second[n+1];
                        neighbor_data.second[n+1] = neighbor_data.second[n];
                        neighbor_data.second[n] = temp;
                    }
                }
            }
            // for (auto data : neighbor_data.second) {
            //     if (data.name.empty()) {
            //         std::cout  << "data wrong!" <<std::endl;
            //         exit(1);
            //     }
            // }
            // std::cout  << " *";
            // sort(neighbor_data.second.begin(), neighbor_data.second.end(), cmp_dist);
            std::cout  << "* ";
            neighbors[i].emplace_back(neighbor_data.first);
        }
        std::cout  <<std::endl;
    }
}

float KMeans::ComputeConnectScore(const int point_id,
    const VecPoint_t& tar_pntcloud,
    const std::vector<std::vector<int>>& connection,
    const std::unordered_set<int>& const_ids){
    std::unordered_set<int> grp_point_ids;
    for (int pnt_id = 0; pnt_id < tar_pntcloud.size(); pnt_id++){
        grp_point_ids.emplace(tar_pntcloud[pnt_id].id);
    }
    const auto& neighbor_point_ids = connection[point_id];
    const int number_neighbor_points = neighbor_point_ids.size();
    if (number_neighbor_points < 1){
        return 0.0;
    }
    float point_score = 0;
    float all_score = 0;
    int num_connect = 0;
    for (int id = 0; id < number_neighbor_points; id++){
        auto neighbor_point_id = neighbor_point_ids[id];
        if (const_ids.find(neighbor_point_id) != const_ids.end()){
            continue;
        }
        if (grp_point_ids.find(neighbor_point_id) != grp_point_ids.end()){
            point_score += 1 + (float)(number_neighbor_points - id - 1)/number_neighbor_points;
        }
        all_score += 1 + (float)(number_neighbor_points - id - 1)/number_neighbor_points;
        if (++num_connect > 10){
            break;
        }
    }

    return (point_score) / all_score;
    // return 0.5;
}

bool KMeans::FilterWeakConnect(std::unordered_set<int>& filter_point_id, const int grp_id,
    const std::vector<std::vector<int>>& connection, const std::unordered_set<int>& const_ids, 
    const float score_thr){
    const int num_pnt = m_grp_pntcloud[grp_id].size();
    filter_point_id.clear();
    filter_point_id.reserve(num_pnt);

#if 1
    std::mutex filter_mutex_;
    auto Filter = [&](int star_id, int end_id){
        for (int pnt_id = star_id; pnt_id < end_id; pnt_id++){
            int point_id = m_grp_pntcloud[grp_id][pnt_id].id;
            float point_score = ComputeConnectScore(
                point_id, m_grp_pntcloud[grp_id], connection, const_ids);
            if(point_score < score_thr){
                std::unique_lock<std::mutex> filter_lock(filter_mutex_);
                filter_point_id.insert(pnt_id);
            }
        }
    };

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    thread_pool.reset(new ThreadPool(num_eff_threads));
    int num_pnt_thread = (num_pnt + num_eff_threads -1) / num_eff_threads;
    for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        int start_idx = thread_idx*num_pnt_thread;
        int end_idx = std::min((thread_idx+1)*num_pnt_thread, num_pnt);
        thread_pool->AddTask(Filter, start_idx, end_idx);
    }
    thread_pool->Wait();
#else
    for (int pnt_id = 0; pnt_id < m_grp_pntcloud[grp_id].size(); pnt_id++){
        int point_id = m_grp_pntcloud[grp_id][pnt_id].id;
        float point_score = ComputeConnectScore(point_id, grp_id, connection, const_ids);
        if(point_score < score_thr){
            filter_point_id.insert(filter_point_id.begin(), pnt_id);
        }
    }
#endif
    return true;
}

bool KMeans::SortConnectScore(std::vector<std::pair<int, float>>& points_score, 
    const std::vector<std::vector<int>>& connection, 
    const std::unordered_set<int>& const_ids,
    const VecPoint_t& tar_pntcloud, const VecPoint_t& pntcloud){
    
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    thread_pool.reset(new ThreadPool(num_eff_threads));

    //compute tar_pntcloud center
    Tuple tar_center;
    size_t pnt_num_in_tar = tar_pntcloud.size();
    float x = 0, y = 0, z = 0;
    for (size_t i = 0; i < pnt_num_in_tar; i++) {
        x += tar_pntcloud[i].location[0];
        y += tar_pntcloud[i].location[1];
        z += tar_pntcloud[i].location[2];
    }
    x /= pnt_num_in_tar;
    y /= pnt_num_in_tar;
    z /= pnt_num_in_tar;

    tar_center.location[0] = x;
    tar_center.location[1] = y;
    tar_center.location[2] = z;

    // compute pont dist
    const int pntcloud_size = pntcloud.size();
    std::vector<float> points_dist(pntcloud_size);
    float max_dist = 0;
    float min_dist = __FLT_MAX__;

    auto Dist = [&](int star_id, int end_id){
        for(int pnt_id = star_id; pnt_id < end_id; pnt_id++){
            points_dist[pnt_id] = DistBetweenPoints(tar_center, pntcloud[pnt_id]);
            if (points_dist[pnt_id] > max_dist){
                max_dist = points_dist[pnt_id];
            }
            if (points_dist[pnt_id] < min_dist){
                min_dist = points_dist[pnt_id];
            }
        }
    };

    int num_pnt_thread = (pntcloud_size + num_eff_threads -1) / num_eff_threads;
    for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        int start_idx = thread_idx*num_pnt_thread;
        int end_idx = std::min((thread_idx+1)*num_pnt_thread, pntcloud_size);
        thread_pool->AddTask(Dist, start_idx, end_idx);
    }
    thread_pool->Wait();

    points_score.clear();
    points_score.resize(pntcloud_size);

    auto Score = [&](int star_id, int end_id){
        for (int pnt_id = star_id; pnt_id < end_id; pnt_id++){
            int point_id = pntcloud[pnt_id].id;
            float connect_score = ComputeConnectScore(
                point_id, tar_pntcloud, connection, const_ids);

            float dist_score =  exp((max_dist - points_dist[pnt_id]) / (max_dist - min_dist));
            float point_score = connect_score < 0.02 ? 0.1 * dist_score : connect_score + dist_score * 0.36;
            // float point_score = connect_score;
            points_score[pnt_id] = std::pair<int, float>(pnt_id, point_score);
        }
    };

    for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        int start_idx = thread_idx*num_pnt_thread;
        int end_idx = std::min((thread_idx+1)*num_pnt_thread, pntcloud_size);
        thread_pool->AddTask(Score, start_idx, end_idx);
    }
    thread_pool->Wait();

    std::sort(points_score.begin(),points_score.end(), 
             [](const std::pair<int, float> & dist1,
             const std::pair<int, float> & dist2) {
             return dist1.second > dist2.second;});

    return true;
}

bool KMeans::SortDistScore(std::vector<std::pair<int, float>>& points_score, 
    const std::unordered_set<int>& const_ids,
    const VecPoint_t& tar_pntcloud, const VecPoint_t& pntcloud){
    
    const int num_eff_threads = GetEffectiveNumThreads(-1);
    thread_pool.reset(new ThreadPool(num_eff_threads));

    //compute tar_pntcloud center
    Tuple tar_center;
    size_t pnt_num_in_tar = tar_pntcloud.size();
    float x = 0, y = 0, z = 0;
    for (size_t i = 0; i < pnt_num_in_tar; i++) {
        x += tar_pntcloud[i].location[0];
        y += tar_pntcloud[i].location[1];
        z += tar_pntcloud[i].location[2];
    }
    x /= pnt_num_in_tar;
    y /= pnt_num_in_tar;
    z /= pnt_num_in_tar;

    tar_center.location[0] = x;
    tar_center.location[1] = y;
    tar_center.location[2] = z;

    // compute pont dist
    const int pntcloud_size = pntcloud.size();
    std::vector<float> points_dist(pntcloud_size);
    float max_dist = 0;
    float min_dist = __FLT_MAX__;
    
    auto Dist = [&](int star_id, int end_id){
        for(int pnt_id = star_id; pnt_id < end_id; pnt_id++){
            points_dist[pnt_id] = DistBetweenPoints(tar_center, pntcloud[pnt_id]);
            if (points_dist[pnt_id] > max_dist){
                max_dist = points_dist[pnt_id];
            }
            if (points_dist[pnt_id] < min_dist){
                min_dist = points_dist[pnt_id];
            }
        }
    };

    int num_pnt_thread = (pntcloud_size + num_eff_threads -1) / num_eff_threads;
    for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        int start_idx = thread_idx*num_pnt_thread;
        int end_idx = std::min((thread_idx+1)*num_pnt_thread, pntcloud_size);
        thread_pool->AddTask(Dist, start_idx, end_idx);
    }
    thread_pool->Wait();

    points_score.clear();
    points_score.resize(pntcloud_size);
    
    // compute score
    auto Score = [&](int star_id, int end_id){
        for (int pnt_id = star_id; pnt_id < end_id; pnt_id++){
            int point_id = pntcloud[pnt_id].id;
            float dist_score =  exp((max_dist - points_dist[pnt_id]) / (max_dist - min_dist));
            points_score[pnt_id] = std::pair<int, float>(pnt_id, dist_score);
        }            
    };

    for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        int start_idx = thread_idx*num_pnt_thread;
        int end_idx = std::min((thread_idx+1)*num_pnt_thread, pntcloud_size);
        thread_pool->AddTask(Score, start_idx, end_idx);
    }
    thread_pool->Wait();
    
    std::sort(points_score.begin(),points_score.end(), 
             [](const std::pair<int, float> & dist1,
             const std::pair<int, float> & dist2) {
             return dist1.second > dist2.second;});

    return true;
}

bool KMeans::UpdateGrpIds(const VecPoint_t& pntcloud, std::unordered_set<int>& const_ids){
    for (auto &pnt : pntcloud){
        const_ids.insert(pnt.id);
    }
    return true;
}

bool KMeans::SameSizeClusterWithConnection(const std::vector<std::vector<int>> connection){
    if (m_k * fixed_size < mv_pntcloud.size()){
        std::cout << "m_k * fix_size < mv_pntcloud.size()" << std::endl;
        return false;
    }

    if ( m_k > 1 && (m_k - 1) * fixed_size > mv_pntcloud.size()){
        std::cout << "(m_k-1) * fix_size > mv_pntcloud.size()" << std::endl;
        return false;
    }

    if (max_point_size < fixed_size){
        std::cout << "param error: max_point_size < fixed_size" << std::endl;
        max_point_size = fixed_size;
    }

    std::vector<VecPoint_t> m_grp_same_size;
    VecPoint_t mv_same_size_center;

    MovetoCenter();
    std::unordered_set<int> const_ids;
    const_ids.reserve(mv_pntcloud.size());

    int k = m_k;
    while(m_k > 0){
        if (m_k > 1){
            PlusCluster();
    	    ComputeGroupCenter();
            
            mv_pntcloud.clear();
            // VecPoint_t cluster_same_size(fixed_size + 1);

            // Calculate numMem
            Eigen::Vector3d mean_location = Eigen::Vector3d::Zero();
            for (int center_idx = 0; center_idx < m_k; center_idx++){
                mean_location += mv_center[center_idx].location;
            }
            mean_location /= m_k;

            // find the furthest center
            int max_dist_idx = 0;
            double max_dist = 0;
            for (int center_idx = 0; center_idx < m_k; center_idx++){
                double dist_center = 
                    (mv_center[center_idx].location - mean_location).norm();
                if (dist_center > max_dist){
                    max_dist = dist_center;
                    max_dist_idx = center_idx;
                }
            }


            for (int center_idx = 0; center_idx < m_k; center_idx++){
                if (center_idx == max_dist_idx){
                    continue;
                }
                mv_pntcloud.insert(mv_pntcloud.end(), 
                                m_grp_pntcloud[center_idx].begin(), 
                                m_grp_pntcloud[center_idx].end());
            }
            // {
            //     std::string save_path = "./" + std::to_string(m_k) + "-ori.ply";
            //     std::vector<VecPoint_t> save_point;
            //     save_point.push_back(m_grp_pntcloud[max_dist_idx]);
            //     // save_point.push_back(mv_pntcloud);
            //     WritePointCloud(save_path, save_point);
            // }

            // fill points
            std::cout << "Cluster id " << m_grp_same_size.size() << "fix size " << fixed_size
                      << "\tori " 
                      << m_grp_pntcloud[max_dist_idx].size() << " points,"
                      << "mv_pntcloud:  " << mv_pntcloud.size() << std::endl;
            if(m_grp_pntcloud[max_dist_idx].size() < 1.5 * fixed_size) {
                std::vector<std::pair<int, float>> grp_dist;
                // SortConnectScore(grp_dist, max_dist_idx, connection, const_ids, mv_pntcloud);
                SortDistScore(grp_dist, const_ids, m_grp_pntcloud[max_dist_idx], mv_pntcloud);
                const int num_fill = 1.5 * fixed_size - m_grp_pntcloud[max_dist_idx].size();
                for (int i = 0; i < num_fill; i++){
                    m_grp_pntcloud[max_dist_idx].push_back(mv_pntcloud.at(grp_dist.at(i).first));
                    mv_pntcloud.at(grp_dist.at(i).first).dist = -1.0;
                    // std::cout << " " << i ;
                }

                for(auto it = mv_pntcloud.begin(); it != mv_pntcloud.end(); it++){
                    if(it->dist < -0.01){
                        it = mv_pntcloud.erase(it);
                        it--;
                        if(it == mv_pntcloud.end()) break;
                    }
                }
            }

            // {
            //     std::string save_path = "./" + std::to_string(m_k) + "-fill.ply";
            //     std::vector<VecPoint_t> save_point;
            //     save_point.push_back(m_grp_pntcloud[max_dist_idx]);
            //     // save_point.push_back(mv_pntcloud);
            //     WritePointCloud(save_path, save_point);
            // }

            //filter weak point
            bool filter_flag = true;
            float score_threshold = 0.3;
            VecPoint_t grp_pointcloud;
            std::cout << "\t fill " 
                      << m_grp_pntcloud[max_dist_idx].size() << " points,"
                      << "mv_pntcloud:  " << mv_pntcloud.size() << std::endl;
            while(filter_flag){
                grp_pointcloud.clear();
                std::unordered_set<int> filter_point_ids;
                FilterWeakConnect(filter_point_ids, max_dist_idx, connection, 
                                  const_ids, score_threshold);

                if (filter_point_ids.size() > m_grp_pntcloud[max_dist_idx].size() * 0.3){
                    // std::cout << "score_threshold: " << score_threshold << " " 
                    //           << filter_point_ids.size() << " vs " 
                    //           << m_grp_pntcloud[max_dist_idx].size() * 0.3 << std::endl;
                    score_threshold /= 1.4;
                    if (score_threshold < 0.05){
                        filter_flag = false;
                    }
                    continue;
                }

                for(int i = 0; i < m_grp_pntcloud[max_dist_idx].size(); i++){
                    if(filter_point_ids.find(i) == filter_point_ids.end()){
                        grp_pointcloud.push_back(m_grp_pntcloud[max_dist_idx].at(i));
                    } else {
                        mv_pntcloud.push_back(m_grp_pntcloud[max_dist_idx].at(i));
                    }
                }
                if (filter_point_ids.size() < std::max(10, (int)(fixed_size * 0.01)) ||
                    (m_grp_pntcloud[max_dist_idx].size() < fixed_size * 0.7) ||
                    score_threshold < 0.1){
                    filter_flag = false;
                }
                m_grp_pntcloud[max_dist_idx] = grp_pointcloud;
            }
            std::cout << "\t filtered " 
                      << m_grp_pntcloud[max_dist_idx].size() << " points"
                      << std::endl;
            if (m_grp_pntcloud[max_dist_idx].size() == 0){
                continue;
            }

            // {
            //     std::string save_path = "./" + std::to_string(m_k) + "-filtered.ply";
            //     std::vector<VecPoint_t> save_point;
            //     save_point.push_back(m_grp_pntcloud[max_dist_idx]);
            //     // save_point.push_back(mv_pntcloud);
            //     WritePointCloud(save_path, save_point);
            // }

            // whether we recurit or discard points
            VecPoint_t grp_same_size;
            if (m_grp_pntcloud[max_dist_idx].size() > fixed_size){
                int num_discard = m_grp_pntcloud[max_dist_idx].size() - fixed_size;

                std::vector<std::pair<int, float>> grp_dist;
                SortConnectScore(grp_dist, connection, const_ids, mv_pntcloud, m_grp_pntcloud[max_dist_idx]);

                // remove the point from center_id
                for (int id = m_grp_pntcloud[max_dist_idx].size() -1; 
                     id >= m_grp_pntcloud[max_dist_idx].size() - fixed_size; id--){
                    grp_same_size.push_back(
                        m_grp_pntcloud[max_dist_idx].at(grp_dist[id].first));
                    // std::cout << "id: " <<  id << " " << grp_dist[id].second << std::endl;
                }

                // update mv_pntcloud
                for (int id = m_grp_pntcloud[max_dist_idx].size() - fixed_size - 1; 
                     id >= 0; id--){
                    mv_pntcloud.push_back(
                        m_grp_pntcloud[max_dist_idx].at(grp_dist[id].first));
                }
            }else if (m_grp_pntcloud[max_dist_idx].size() == fixed_size){
                grp_same_size = m_grp_pntcloud[max_dist_idx];
            }else if (m_grp_pntcloud[max_dist_idx].size() < fixed_size){
                struct GrpIdx{
                    int center_id;
                    int pnt_id;
                };
                
                std::vector<std::pair<int, float>> grp_dist;
                int num_iter = 0;
                float min_score = 1.2;
                while(m_grp_pntcloud[max_dist_idx].size() < fixed_size){
                    grp_dist.clear();
                    min_score = std::min (min_score - 0.02, 0.8);

                    SortConnectScore(grp_dist, connection, const_ids, 
                        m_grp_pntcloud[max_dist_idx], mv_pntcloud);
                    num_iter++;
                    if (num_iter > 20){
                        std::cout << "Cluster id " << m_grp_same_size.size()
                                  << " Weak connection " 
                                  << m_grp_pntcloud[max_dist_idx].size() << " points"
                                  << std::endl;
                        // min_score = 1.0;
                        break;
                    }

                    for (int i = 0; i < fixed_size - m_grp_pntcloud[max_dist_idx].size(); i++){
                        if (grp_dist.at(i).second < min_score){
                            break;
                        }
                        m_grp_pntcloud[max_dist_idx].push_back(mv_pntcloud.at(grp_dist.at(i).first));
                        mv_pntcloud.at(grp_dist.at(i).first).dist = -1.0;
                    }

                    for(auto it = mv_pntcloud.begin(); it != mv_pntcloud.end(); it++){
                        if(it->dist < -0.01){
                            it = mv_pntcloud.erase(it);
                            it--;
                            if(it == mv_pntcloud.end()) break;
                        }
                    }
                }
                grp_same_size = m_grp_pntcloud[max_dist_idx];
                // std::cout << "mv_pntcloud size: " << mv_pntcloud.size() << std::endl;
            }
            std::cout << "\t end " 
                      << grp_same_size.size() << " points"
                      << std::endl;

            UpdateGrpIds(grp_same_size, const_ids);
            
            // {
            //     std::string save_path = "./" + std::to_string(m_k) + "-end.ply";
            //     std::vector<VecPoint_t> save_point;
            //     save_point.push_back(grp_same_size);
            //     // save_point.push_back(mv_pntcloud);
            //     WritePointCloud(save_path, save_point);
            // }

            m_grp_same_size.push_back(grp_same_size);
            mv_same_size_center.push_back(mv_center[max_dist_idx]);
            m_grp_pntcloud.clear();
            mv_center.clear();

            // std::cout << "mv_pntcloud: " << mv_pntcloud.size() << std::endl;
            // std::cout << "grp_same_size" << grp_same_size.size() << std::endl;
            int num_points = mv_pntcloud.size();
            if (num_points / (m_k - 1) <= max_point_size){
                m_k--;
            } 
            fixed_size = num_points / m_k + 1;
            m_grp_pntcloud.resize(m_k);
        } else if (m_k == 1){
            Cluster();
            m_grp_same_size.push_back(m_grp_pntcloud.at(0));
            mv_same_size_center.push_back(mv_center.at(0));
            m_grp_pntcloud.clear();
            mv_center.clear();
            m_k--;
        }
    }

    m_grp_pntcloud.swap(m_grp_same_size);
    mv_center.swap(mv_same_size_center);
    m_k = m_grp_pntcloud.size();

    ComputeGroupCenter();
    std::cout << "m_grp_pntcloud size: ";
    for (int i = 0; i < m_grp_pntcloud.size(); i++){
        std::cout << m_grp_pntcloud.at(i).size() << " ";;
    }
    std::cout << std::endl;

    return true;
}

bool cmp_score(Tuple& a, Tuple& b) { return a.dist > b.dist; }

void KMeans::FindNeighborsAndCommonPointsWithConnection(
    std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
    std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<int>>& connection) {
    neighbors_points.resize(m_k);
    neighbors.resize(m_k);
    // find common images.
    for (size_t i = 0; i < m_k; i++) {
        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            int neighbor;
            double max_score = -1;
            for (size_t k = 0; k < m_k; k++) {
                if (i == k) {
                    continue;
                }
                int point_id = m_grp_pntcloud[i][j].id;
                float score = ComputeConnectScore(point_id, m_grp_pntcloud[k], connection);
                if (score - max_score > 0.000001) {
                    max_score = score;
                    neighbor = k;
                }
            }
            m_grp_pntcloud[i][j].dist = max_score;
            neighbors_points[i][neighbor].emplace_back(m_grp_pntcloud[i][j]);
        }
    }
    // find neighbors.
    for (size_t i = 0; i < m_k; i++) {
        // neighbors[i].reserve(neighbors_points[i].size());
        std::cout << i << " neighbors:";
        for (auto& neighbor_data : neighbors_points[i]) {
            sort(neighbor_data.second.begin(), neighbor_data.second.end(), cmp_score);
            neighbors[i].emplace_back(neighbor_data.first);
            std::cout << " " << neighbor_data.first;
        }
        std::cout << std::endl;
    }
}


void KMeans::FindNeighborsAndCommonPointsWithConnection_AllPoints(
    std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
    std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<int>>& connection) {
    neighbors_points.resize(m_k);
    neighbors.resize(m_k);
    // find common images.

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    thread_pool.reset(new ThreadPool(num_eff_threads));
    double max_dist = 0;
    for (size_t i = 0; i < m_k - 1; i++) {
        for (size_t j = i + 1; j < m_k; j++) {
            double dist = DistBetweenPoints(mv_center[i], mv_center[j]);
            if (max_dist < dist){
                max_dist = dist;
            }
        }
    }

    for (size_t i = 0; i < m_k; i++) {
        int num_grp_pnt = m_grp_pntcloud[i].size();
        std::unordered_map<int, int> num_k;
        for (size_t k = 0; k < m_k; k++) {
            if (i == k) {
                continue;
            }
            neighbors_points[i][k].resize(num_grp_pnt);
            num_k[k] = 0;
        }
#if 1
        std::mutex score_mutex_;
        auto Score = [&](int star_id, int end_id){
            for (size_t j = star_id; j < end_id; j++) {
                int neighbor;
                double max_score = -1;
                double min_dist = __DBL_MAX__;
                for (size_t k = 0; k < m_k; k++) {
                    if (i == k) {
                        continue;
                    }
                    int point_id = m_grp_pntcloud[i][j].id;
                    float score = ComputeConnectScore(point_id, m_grp_pntcloud[k], connection);
                    for (size_t m = 0; m < m_grp_pntcloud[k].size(); m++) {
                        double dist = DistBetweenPoints(m_grp_pntcloud[i][j], m_grp_pntcloud[k][m]);
                        if (min_dist - dist > 0.000001) {
                            min_dist = dist;
                        }
                    }
                    // score *= exp((max_dist - min_dist) / (max_dist));
                    float dist_score = exp((max_dist - min_dist) / (max_dist));
                    score = score < 0.02 ? 0.1 * dist_score : score + dist_score * 0.36;
                    if (score - max_score > 0.000001) {
                        max_score = score;
                        neighbor = k;
                    }
                }
                m_grp_pntcloud[i][j].dist = max_score;
                {
                    std::unique_lock<std::mutex> score_lock(score_mutex_);
                    neighbors_points[i][neighbor][num_k[neighbor]++] = m_grp_pntcloud[i][j];
                }
            }
        };

        int num_pnt_thread = (num_grp_pnt + num_eff_threads - 1) / num_eff_threads;
        for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx){
            int start_idx = thread_idx * num_pnt_thread;
            int end_idx = std::min((thread_idx+1)*num_pnt_thread, num_grp_pnt);
            thread_pool->AddTask(Score, start_idx, end_idx);
        }
        thread_pool->Wait();

        for (size_t k = 0; k < m_k; k++) {
            if (k == i){
                continue;
            }
            // std::cout << "num_k.at(k)" << k << " " << num_k.at(k) << std::endl;
            if (num_k[k] == 0){
                neighbors_points[i].erase(k);
            } else {
                neighbors_points[i][k].resize(num_k[k]);
            }
        }
        // std::cout << std::endl;
#else
        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            int neighbor;
            double max_score = -1;
            double min_dist = __DBL_MAX__;
            for (size_t k = 0; k < m_k; k++) {
                if (i == k) {
                    continue;
                }
                int point_id = m_grp_pntcloud[i][j].id;
                float score = ComputeConnectScore(point_id, m_grp_pntcloud[k], connection);
                for (size_t m = 0; m < m_grp_pntcloud[k].size(); m++) {
                    double dist = DistBetweenPoints(m_grp_pntcloud[i][j], m_grp_pntcloud[k][m]);
                    if (min_dist - dist > 0.000001) {
                        min_dist = dist;
                    }
                }
                // double max_dist = DistBetweenPoints(m_grp_pntcloud[i][j], mv_center[k]);
                score *= exp((max_dist - min_dist) / (max_dist));
                if (score - max_score > 0.000001) {
                    max_score = score;
                    neighbor = k;
                }
            }
            m_grp_pntcloud[i][j].dist = max_score;
            neighbors_points[i][neighbor].emplace_back(m_grp_pntcloud[i][j]);
            // neighbors_points[i][neighbor][num_k.at(neighbor)] = m_grp_pntcloud[i][j];
            num_k[neighbor] = num_k[neighbor] + 1;
        }

        for (size_t k = 0; k < m_k; k++) {
            if (k == i){
                continue;
            }
            std::cout << "num_k.at(k)" << k << " " << num_k[k] << std::endl;
            std::cout << "neighbors_points[i][neighbor] " << i << "-" << k << " " << neighbors_points[i][k].size() << std::endl;
            // neighbors_points[i][k].resize(num_k.at(k));
        }
        std::cout << std::endl;
#endif
    }
    // find neighbors.
    for (size_t i = 0; i < m_k; i++) {
        // neighbors[i].reserve(neighbors_points[i].size());
        std::cout << i << " neighbors:";
        for (auto& neighbor_data : neighbors_points[i]) {
            neighbor_data.second.shrink_to_fit();
            if (neighbor_data.second.empty()) {
                std::cout  << "neighbor_data.second.empty()" <<std::endl;
                exit(1);
            }
            sort(neighbor_data.second.begin(), neighbor_data.second.end(), cmp_score);
            neighbors[i].emplace_back(neighbor_data.first);
            std::cout << " " << neighbor_data.first;
        }
        std::cout << std::endl;
    }
}

bool KMeans::WritePointCloud(std::string SavePlyPath){

    cv::RNG rng(12345);
    std::vector<PlyPoint> locations;
    std::vector<PlyPoint> color_map;
    std::vector<PlyPoint> mv_locations;

    std::vector<Eigen::Vector3i> color_k(m_k);
    for (size_t i = 0; i < m_k; i++) {
        int r_color = rng.uniform(0, 255);
        int g_color = rng.uniform(0, 255);
        int b_color = rng.uniform(0, 255);

        color_k[i][0] = r_color;
        color_k[i][1] = g_color;
        color_k[i][2] = b_color;

        for (size_t j = 0; j < m_grp_pntcloud[i].size(); j++) {
            PlyPoint point;
            point.x = float(m_grp_pntcloud[i][j].location[0]);
            point.y = float(m_grp_pntcloud[i][j].location[1]);
            point.z = float(m_grp_pntcloud[i][j].location[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            locations.emplace_back(point);
        }

        PlyPoint color;
        color.x = i;
        color.y = 0;
        color.z = 0;
        color.r = r_color;
        color.g = g_color;
        color.b = b_color;
        color_map.emplace_back(color);

        PlyPoint nv_loc;
        nv_loc.x = float(mv_center[i].location[0]);
        nv_loc.y = float(mv_center[i].location[1]);
        nv_loc.z = float(mv_center[i].location[2]);
        nv_loc.r = r_color;
        nv_loc.g = g_color;
        nv_loc.b = b_color;
        mv_locations.emplace_back(nv_loc);
    }

    sensemap::WriteBinaryPlyPoints(SavePlyPath, locations, false, true);
    sensemap::WriteBinaryPlyPoints(SavePlyPath + "-color.ply", color_map, false, true);
    sensemap::WriteBinaryPlyPoints(SavePlyPath + "-center.ply", mv_locations, false, true);
    return true;
}

bool KMeans::WritePointCloud(std::string SavePlyPath, 
                             std::vector<VecPoint_t> grp_pntcloud){

    cv::RNG rng(12345);
    std::vector<PlyPoint> locations;

    std::vector<Eigen::Vector3i> color_k(m_k);
    for (size_t i = 0; i < grp_pntcloud.size(); i++) {
        int r_color = rng.uniform(0, 255);
        int g_color = rng.uniform(0, 255);
        int b_color = rng.uniform(0, 255);

        color_k[i][0] = r_color;
        color_k[i][1] = g_color;
        color_k[i][2] = b_color;

        for (size_t j = 0; j < grp_pntcloud[i].size(); j++) {
            PlyPoint point;
            point.x = float(grp_pntcloud[i][j].location[0]);
            point.y = float(grp_pntcloud[i][j].location[1]);
            point.z = float(grp_pntcloud[i][j].location[2]);
            point.r = r_color;
            point.g = g_color;
            point.b = b_color;
            locations.emplace_back(point);
        }
    }

    sensemap::WriteBinaryPlyPoints(SavePlyPath, locations, false, true);
    return true;
}
}  // namespace sensemap
