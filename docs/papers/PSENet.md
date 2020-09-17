

论文题目：
**Shape Robust Text Detection with Progressive Scale Expansion Network**

概括：该paper提出Progressive Scale Expansion network 的思想,以语义分割的

论文主要思想：
* 语义分割的思路做文字检测
* 提出Progressive Scale Expansion network，以最小scale kernel作为文本分割实例的起点（有利于分开不同的文本实例）。然后利用其他不同尺寸的kernel图，逐步扩增文本实例的边缘，以确定文本实例的边缘。


text kernel：跟原始文本groun truth 实例拥有相似的形状但是大小scale不一样的文本实例。
PSENet中， $text kernel scale  \in (0.4,1)$ 

整体架构如图所示：
![](http://image.haibin.online/2020-09-09,19:57:02.jpg)

* backbone网络提取4个阶段的视觉语义特征
* 类似FPN的网络进行特征融合
* FPN特征融合后，得到最终特征图F

特征图F的size为（batch_size,n,w,h）
n为分割图的数量。产生的n个分割特征图（S1，S2，S3···Sn），每一个$S_{i}$都以一定比例的比例作为语义分割中实例的mask值。$S_{1}$作为比例最小的文本实例结果，$S_{n}$则表示文本GT原始大小的分割mask（the maximal kernels)。

从$S_{1}$出发，将所有文本实例的内核逐步扩展为$S_{n}$的完整形状，并最后获得最终检测结果$R$


### 扩张算法详解

![](http://image.haibin.online/2020-09-09,20:55:12.jpg)

如图所示，3个分割结果$S = \{S_{1},S_{2},S_{3}\}$
1. 基于最小的kernels’map $S_{1}$,可以得到4个不同的连通区域（connected components $C=\{c_{1},c_{2},c_{3},c_{4}\}$）如b所示.
2. $C=\{c_{1},c_{2},c_{3},c_{4}\}$作为本张图片预测的所有文本实例的中心部分。从此各个连通区域的核心出发，利用广度优先搜索的策略，在S2中，如果pixel $p$在文本实例$c_{1}$的周围，且$p$在$S_{2}$中属于文本区域像素，则该像素$p$则扩张成为$c_{1}$的边界。（整个过程如图（g）所示）
3. 直到扩张到S3，最终得到文本实例的最终边界。

扩展基于广度优先搜索算法，该算法从多个内核的像素开始，并迭代合并相邻的文本像素。在实践中，某个像素可能会属于多个文本实例的边界，此时会产生冲突。解决冲突的办法是采用先到先得的策略由单个kernel进行合并。


代码详解：
本部分代码来源于[PSENet.pytorch](https://github.com/WenmuZhou/PSENet.pytorch.git) 仓库：

pse.cpp

```cpp
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
            //将next_q再进行一次扩展
            std::swap(q, next_q);
        }
        return res;
    }
}

```

伪代码如图所示：

![](http://image.haibin.online/2020-09-09,21:26:11.jpg)



### label生成

如上面过程描述，我们要生成$S_{1}-S_{n}$张文本实例图，则需要n张gt label mask参与loss计算。在训练过程中，这些 ground truth labels mask 可以缩小 通过原始文本实例的方式得到。

![](http://image.haibin.online/2020-09-09,21:30:17.jpg)

多边形缩小采用 Vatti clipping algorithm 算法进行原始多边形的$p_{n}$的缩放得到$P_{i}$通过距离$d_{i}$

$$d_{i}=\frac{Area(p_{n})*(1-r_{i}^{2})}{Permimeter(P_{n})}$$

$$r_{i} = 1-\frac{(1-m)*(n-i)}{n-1}$$

其中m为最小尺寸比例，取值范围为(0,1],默认取0.4

生成代码如下所示：
```python
@PIPELINES.register_module
class GenerateTrainMaskPSE(object):
    """
        shrink_ratio: gt收缩的比例
        vatli clipping 算法收缩 training textregion
        """

    def __init__(self, result_num:int=6,m:float=0.5):
        self.n = result_num
        self.m = m

    def __call__(self,data:dict):
        h, w, c = data["img"].shape
        text_polys = data["gt_polys"]
        text_tags = data["gt_tags"]
        training_mask = np.ones((h, w), dtype=np.uint8)
        score_maps = []

        for si in range(1,self.n+1):
            score_map, training_mask = self.generate_rbox((h, w), text_polys, text_tags, training_mask, si,self.n,self.m)
            score_maps.append(score_map)
        score_maps = np.array(score_maps, dtype=np.float32)
        data["gt"] = score_maps
        data["mask"] = training_mask
        return data

    def generate_rbox(self,im_size,text_polys, text_tags, training_mask, i, n, m):
        """
        生成mask图，白色部分是文本，黑色是背景
        :param im_size: 图像的h,w
        :param text_polys: 框的坐标
        :param text_tags: 标注文本框是否参与训练
        :param training_mask: 忽略标注为 DO NOT CARE 的矩阵
        :return: 生成的mask图
        """
        h, w = im_size
        score_map = np.zeros((h, w), dtype=np.uint8)
        for poly, tag in zip(text_polys, text_tags):
            try:
                poly = poly.astype(np.int)
                r_i = 1 - (1 - m) * (n - i) / (n - 1)
                d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
                pco = pyclipper.PyclipperOffset()

                pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked_poly = np.array(pco.Execute(-d_i))
                cv2.fillPoly(score_map, shrinked_poly, 1)
                # 制作mask
                # rect = cv2.minAreaRect(shrinked_poly)
                # poly_h, poly_w = rect[1]

                # if min(poly_h, poly_w) < 10:
                #     cv2.fillPoly(training_mask, shrinked_poly, 0)
                if not tag:
                    cv2.fillPoly(training_mask, shrinked_poly, 0)
                # 闭运算填充内部小框
                # kernel = np.ones((3, 3), np.uint8)
                # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
            except Exception as e:
                continue
        return score_map, training_mask
```


生成结果如图所示，s1-s6 文字实例区域由小到大排列
![](http://image.haibin.online/2020-09-09,21:51:24.jpg)


### loss 设计

PSENet的loss 由两部分组成
* $Lc$代表了 完整文本实例的loss complete text instances 的loss
* $Ls$代表了缩小部分文本kernels的loss
用系数$\lambda$ 这两部分loss的比例 $\lambda \in(0,1]$

$$L=\lambda L_{c}+(1-\lambda) L_{s}$$


通常情况下文本区域仅占整个图像很少的一部分，使用binary cross entropy 二进制交叉熵的话会使得神经网络的的预测偏向于非文本区域，所以PSENet 采用Dice loss，其Dice coefficient定义为：


$$D\left(S_{i}, G_{i}\right)=\frac{2 \sum_{x, y}\left(S_{i, x, y} \times G_{i, x, y}\right)}{\sum_{x, y} S_{i, x, y}^{2}+\sum_{x, y} G_{i, x, y}^{2}}$$

$S_{i, x, y}$ 为第i个分割结果中第(x,y)个像素的值。$G_{i, x, y}$为第i个groud truth中（x，y）的像素值。

同时，PSENet采用了OHEM，在线困难样本学习，其trainning mask定义为：M
则完整文本实例的loss为：

$$L_{c}=1-D\left(S_{n} \cdot M, G_{n} \cdot M\right)$$


$L{s}$则为剩余缩小文本的loss计算：

其定义为：

$$\begin{aligned}
L_{s}=1-\frac{\sum_{i=1}^{n-1} D\left(S_{i} \cdot W, G_{i} \cdot W\right)}{n-1} & \\
W_{x, y}=\left\{\begin{array}{ll}
1, & \text {if } S_{n, x, y} \geq 0.5 \\
0, & \text { otherwise. }
\end{array}\right.
\end{aligned}$$

其中：$W$定义为非文本区域的mask（即如果该文本被忽略的话即不参与计算）


代码：
```python
class PSELoss(nn.Module):
    def __init__(self, Lambda, ratio=3, reduction='mean'):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = Lambda
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, outputs:torch.Tensor, gt:torch.Tensor, training_masks:torch.Tensor):
        """

        outputs:[batch_size,result_num,w,h]
        labels :[batch_size,result_num,w,h]
        training_masks:[batch_size,w,h]
        """

        ##sn作为gt中 文本框最大的gt图，作为text区域选择
        ##s1 -  sn-1 作为kernel

        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = gt[:, -1, :, :]
        gt_kernels = gt[:, :-1, :, :]

        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)
        
        #最大的Sn作为Lc计算 diceloss
        loss_text = self.dice_loss(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(outputs.device)
        kernels_num = gt_kernels.size()[1]
        ##剩余的kernel计算Ls
        for i in range(kernels_num):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.stack(loss_kernels).mean(0)
        if self.reduction == 'mean':
            loss_text = loss_text.mean()
            loss_kernels = loss_kernels.mean()
        elif self.reduction == 'sum':
            loss_text = loss_text.sum()
            loss_kernels = loss_kernels.sum()

        # loss = self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels
        result = dict(
            loss_text=loss_text ,loss_kernels= loss_kernels,
            loss=self.Lambda * loss_text + (1 - self.Lambda) * loss_kernels
        )
        return result
        # return loss_text, loss_kernels, loss

    def dice_loss(self, input_tensor:torch.Tensor, target:torch.Tensor, mask:torch.Tensor):

        input_tensor = torch.sigmoid(input_tensor)
        input_tensor = input_tensor.contiguous().view(input_tensor.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input_tensor = input_tensor * mask
        target = target * mask

        a = torch.sum(input_tensor * target, 1)
        b = torch.sum(input_tensor * input_tensor, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * 3, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

```



### 总结讨论

(1)方法对比

* regression-based approaches： 文本目标通常以具有特定方向的矩形或四边形的形式表示。 但是，基于回归的方法无法处理具有任意形状的文本实例。

* Segmentation-based approaches：基于语义分割的方法基于像素级分类来定位文本实例。 但是，难以分离彼此接近的文本实例。

(2)kernel是否可以直接作为分类结果？

kernel的目的在于粗略地定位文本实例，并将靠得很近的文本实例分开。但是最小scale的kernel无法覆盖文本实例的完整区域，导致文字区域裁剪不全（会对文字识别造成很明显的错误）
直接使用最大的文本kernel作为结果又不能分离各个文本实例。

（3）最小内核尺寸的选择与内核数量的选择：

个人觉得根据不同数据集选择，比如票据识别中文本实例较多而且目标框较小，这时候就应该选择大一点的最小内核尺寸，避免在缩放过程中较小的gt因为缩放后面积太小而被忽略掉。



