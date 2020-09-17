//
//  pse
//  reference https://github.com/whai362/PSENet/issues/15
//  Created by liuheng on 11/3/19.
//  Copyright © 2019年 liuheng. All rights reserved.
//
#include <queue>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace py = pybind11;

namespace pse{
    //S5->S0, small->big
    std::vector<std::vector<int32_t>> pse(
    py::array_t<int32_t, py::array::c_style> label_map,
    py::array_t<uint8_t, py::array::c_style> Sn,
    int c = 6)
    {
        auto pbuf_label_map = label_map.request();
        auto pbuf_Sn = Sn.request();
        if (pbuf_label_map.ndim != 2 || pbuf_label_map.shape[0]==0 || pbuf_label_map.shape[1]==0)
            throw std::runtime_error("label map must have a shape of (h>0, w>0)");
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];
        if (pbuf_Sn.ndim != 3 || pbuf_Sn.shape[0] != c || pbuf_Sn.shape[1]!=h || pbuf_Sn.shape[2]!=w)
            throw std::runtime_error("Sn must have a shape of (c>0, h>0, w>0)");
        // 结果矩阵 [x,y]=label_i，代表第x,y位置的像素属于第i个文本实例
        std::vector<std::vector<int32_t>> res;
        for (size_t i = 0; i<h; i++)
            res.push_back(std::vector<int32_t>(w, 0));
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_Sn = static_cast<uint8_t *>(pbuf_Sn.ptr);

        std::queue<std::tuple<int, int, int32_t>> q, next_q;

        for (size_t i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            for(size_t j = 0; j<w; j++)
            {
                //获取到最小kernel的中的各个像素的label，先确定最小文本实例
                int32_t label = p_label_map[j];
                if (label>0)
                {
                    //像素入队列，准备进行扩展
                    q.push(std::make_tuple(i, j, label));
                    res[i][j] = label;
                }
            }
        }
        //四个方向
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        // merge from small to large kernel progressively
        for (int i = 1; i<c; i++)
        {
            //get each kernels
            //从S2-Sn，获取每一张kernel图
            auto p_Sn = ptr_Sn + i*h*w;
            while(!q.empty()){
                //get each queue menber in q
                auto q_n = q.front();
                q.pop();
                int y = std::get<0>(q_n);
                int x = std::get<1>(q_n);
                int32_t l = std::get<2>(q_n);
                //store the edge pixel after one expansion
                bool is_edge = true;
                //遍历四个方向上的点
                for (int idx=0; idx<4; idx++)
                {
                    int index_y = y + dy[idx];
                    int index_x = x + dx[idx];
                    //超出边界则不要
                    if (index_y<0 || index_y>=h || index_x<0 || index_x>=w)
                        continue;
                    //该边缘点如果不是文字像素或者已经有label，则放弃
                    if (!p_Sn[index_y*w+index_x] || res[index_y][index_x]>0)
                        continue;
                    //将该边缘点的label设置为中心点的label，并入队等待下一步扩张
                    q.push(std::make_tuple(index_y, index_x, l));
                    res[index_y][index_x]=l;
                    is_edge = false;
                }
                //如果四个方向都跳过，那证明这个点要么是图像边缘的点，要么是文字实例的边缘点，该点加入next_q中
                if (is_edge){
                    next_q.push(std::make_tuple(y, x, l));
                }
            }
            std::swap(q, next_q);
        }
        return res;
    }
}

PYBIND11_MODULE(pse_cpp, m){
    m.def("pse_cpp_f", &pse::pse, " re-implementation pse algorithm(cpp)", py::arg("label_map"), py::arg("Sn"), py::arg("c")=6);
}

