//
// Created by zhou on 2020/9/14.
//

/*
//  reference https://github.com/whai362/pan_pp.pytorch/blob/master/models/post_processing/pa/pa.pyx
//  Copyright © 2020年 zhou. All rights reserved.
*/



#include <queue>
#include <math.h>
#include <map>
#include <algorithm>
#include <vector>
#include "pybind11.h"
#include "numpy.h"
#include "stl.h"
#include "stl_bind.h"
namespace py = pybind11;

namespace pa {

    py::array_t<int32_t> pa_cpp_f(
            py::array_t<float, py::array::c_style> similarity_vectors,
            py::array_t<int32_t, py::array::c_style> label_map,
            py::array_t<int32_t, py::array::c_style> text_components,
            int label_num,
            float min_area = 1
    ) {

        auto pbuf_label_map = label_map.request();


        auto r_cc = text_components.unchecked<2>();
        auto r_similarity_vector  = similarity_vectors.unchecked<3>();
        auto r_label_map = label_map.mutable_unchecked<2>();

        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        //初始化结果
        auto res = py::array_t<int32_t>({h,w});
        auto r_res = res.mutable_unchecked<2>();


        //array 初始化
        for(int i=0;i<h;i++){
            for(int j=0;j<w;j++){
                r_res(i,j) = 0;
            }
        }




        float area_array[label_num] = {0};

        bool flag_array[label_num] = {false};

        int p_array[label_num][2] = {0};

        float  sum_simi_array[label_num][4] = {0};
        float mean_simi_vector[label_num][4] = {0};

        float max_rate = 1024;

        std::vector<std::pair<int, int>> point_vectors;
        point_vectors.reserve(w*h);
        int area_num = 0;
        for (int label_index_i = 1; label_index_i < label_num; label_index_i++) {

            area_num = 0;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if (r_label_map(i, j) == label_index_i) {
                        point_vectors.push_back(std::make_pair(i, j));
                        area_num += 1;
                        for(int k=0;k<4;k++){
                            sum_simi_array[label_index_i][k] += r_similarity_vector(k,i,j);
                        }
                    }
                }
            }

            area_array[label_index_i] = area_num;
            if (area_num < min_area) {
                //某文字核心太小，直接忽略该文字实例
                for(size_t k=0;k<point_vectors.size();k++){
                    r_label_map(point_vectors[k].first,point_vectors[k].second) = 0;
                }
                point_vectors.clear();
                continue;

            }
            //记录第一个点的坐标
            if(!point_vectors.empty()){

                p_array[label_index_i][0] =point_vectors[0].first;
                p_array[label_index_i][1] = point_vectors[0].second;
            }

            for(int k=0;k<4;k++){
                //mean_emb[i] = np.mean(emb[:, ind], axis=1)
                if(!point_vectors.empty()){
                    mean_simi_vector[label_index_i][k] = sum_simi_array[label_index_i][k]/point_vectors.size();
                }

            }
            point_vectors.clear();
            //统计比i之前的文字实例情况，如果两个文字实例面积相差过大，证明该kernel附近的点在聚类的时候要注意聚类距离问题，避免某些框过大或者过小
            for (int label_index_j = 1; label_index_j < label_index_i; label_index_j++) {
                if (area_array[label_index_j] < min_area) {
                    continue;
                }
                //如果i，j两个实例的第一个点坐标不同属于文字像素的话，跳过
                if(r_cc(p_array[label_index_i][0],p_array[label_index_i][1])!=r_cc(p_array[label_index_j][0],p_array[label_index_j][1])){
                    continue;
                }

                float rate = area_array[label_index_i] / area_array[label_index_j];
                //先计算所有的mean_simi,最后flag 为false 置为0，实现了 if 中将 i，j都计算cal的效果


                if (rate < (1.0 / max_rate) || rate > max_rate) {
                    flag_array[label_index_i] = true;
                    flag_array[label_index_j]=true;

                }
            }

        }





        std::queue<std::pair<int, int>> que, next_quq;

        //四个方向
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        int tempx, tempy = 0;
        //points = np.array(np.where(label > 0)).transpose((1, 0);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (r_label_map(i, j) > 0) {
                    que.push(std::make_pair(i, j));
                    r_res(i, j) = r_label_map(i, j);
                }
            }
        }

        int cur_label = 0;


        float norm_sum = 0;
        float norm = 0;
//        while (!que.empty() || !next_quq.empty()) {
        while (!que.empty()) {
            auto cur = que.front();
            que.pop();
//            int y = cur.first;
//            int x = cur.second;

            cur_label = r_res(cur.first, cur.second);
            for (int k = 0; k < 4; k++) {
                tempx = cur.second + dx[k];
                tempy = cur.first + dy[k];
                if (tempx < 0 || tempx > w || tempy < 0 || tempy > h) {
                    continue;
                }
                if (r_cc(tempy, tempx) == 0 || r_res(tempy, tempx) > 0) {
                    continue;
                }
                if (flag_array[cur_label] == 1) {
                    /*
                     * if flag[cur_label] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[cur_label]) > 3:
                continue

                     np.linalg.norm计算二范式

                     */
                    //如果相似向量的二范式距离超过3，则认为不是中心的聚类
                    norm_sum = 0;
                    norm = 0;
                    for (int l = 0; l < 4; l++) {
                        norm_sum += pow(r_similarity_vector(l, tempy, tempx) - mean_simi_vector[cur_label][l],2);
                    }
                    norm = sqrt(norm_sum);
                    if (norm > 3) {
                        continue;
                    }
                }

                que.push(std::make_pair(tempy, tempx));
                r_res(tempy, tempx) = cur_label;
            }

        }
//            std::swap(que, next_quq);
//        }
        return res;
    }
}

PYBIND11_MODULE(pa_cpp,m){
    m.def("pa_cpp_f",&pa::pa_cpp_f," re-implementation pa algorithm(cpp)",
          py::arg("similarity_vectors"),py::arg("label_map"),py::arg("text_components"),
          py::arg("label_num"),py::arg("min_area")=1.0);
}