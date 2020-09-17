

论文题目：
Real-time Scene Text Detection with Differentiable Binarization


基于分割的方法在文本检测中非常流行，因为分割结果可以更准确地描述各种形状的场景文本。
但是二值化的后处理对基于分割的文本检测方法来说是必不可少的，它基于分割结果产生的概率图转化为文本的边界/区域。


本文idea：
* 提出了“可微分二值化”模块，该模块在分割网络中进行自适应像素二值化过程，简化了后处理，提高了文本检测。


做法比较：

![](http://image.haibin.online/2020-09-10,20:42:14.jpg)


原先一些基于分割的做法如蓝色箭头所示：
* 首先，它们设置了固定的阈值，用于将分割网络生成的概率图转换为二进制图像；
*  然后，用一些启发式技术（例如像素聚类）用于将像素分组为文本实例


DB的做法如红色箭头所示：
* 在得到 分割map后，与网络生成的threshold map一次联合做可微分二值化得到二值化图，然后再经过后处理得到最终结果。
* 将二值化操作插入到分段网络中以进行联合优化，通过这种方式，可以自适应地预测图像每个位置的阈值，从而可以将像素与前景和背景完全区分开。 但是，标准二值化函数是不可微分的，因此，我们提出了一种二值化的近似函数，称为可微分二值化（DB），当训练时，该函数完全可微分。

（将一个固定的阈值训练为一个可学习的每个位置的阈值


那么，我们首先看label是如何生成的，网络要学习的目标gt 与 threshold map是怎样的生成和指导网络去训练的；

### label generation

label生成的流程图如下所示：

![](http://image.haibin.online/2020-09-10,21:07:17.jpg)


使用Vatti clipping algorithm 对gt多边形polygon 进行缩放；

offset of shrinking $D$通过原始多边形的周长L和面积A得到：

$$D=\frac{A\left(1-r^{2}\right)}{L}$$

$r$是缩放比例，一般取值为 0.4

这样我们就通过gt polygon形成 缩小版的polygon的gt mask图 probability map（蓝色边界）


以同样的offset D 从多边形polygon $G$ 拓展到$G_{d}$，得到如图中 threshold_map中的绿色边界

threshold_map中由$G_s$ 到$G_d$之间形成了一个文字区域的边界。

一组图来可视化图像生成的结果：
![](http://image.haibin.online/2020-09-10,21:51:57.jpg)


我们可以看到  probability map 的gt是一个完全的0，1 mask ,polygon 的缩小区域为1，其他背景区域为0；

但是在threshold_map文字边框值并非0,1；

使用pycharm的view array 我们能看到threshold_map中文字边框的数值信息：
![](http://image.haibin.online/2020-09-10,22:22:39.jpg)


* 文字最外圈边缘为0.7，靠近中心区域是为0.3的值。（0.3-0.7为预设的阈值最大最小值）。我们可以看到文字边界为阈值最大，然后根据文字实例边缘距离逐渐递减。


知道threshold_map的label值跟gt的值，我们才能更好地去理解“可微分二值化”是如何实现的；


### Differentiable binarization


DB方法整体如图所示：
![](http://image.haibin.online/2020-09-11,16:47:33.jpg)

在分割结果pred 得到 probability_map 跟threshold_map 后，联合做DB得到approximate binary_map。


传统的阈值分割做法为：

$$B_{i, j}=\left\{\begin{array}{ll}
1 & \text { if } P_{i, j}>=t \\
0 & \text { otherwise }
\end{array}\right.$$

$B_{i,j}$代表了probability_map中第i行第j列的概率值。这样的做法是硬性将概率大于某个固定阈值的像素作为文字像素，而不能将阈值作为一个可学习的参数对象（因为阈值分割没办法微分进行梯度回传）



可微分的二值化公式：

$$\hat{B}_{i, j}=\frac{1}{1+e^{-k\left(P_{i, j}-T_{i, j}\right)}}$$

首先，该公式借鉴了sigmod函数的形式（sigmod 函数本身就是将输入映射到0~1之间），所以将概率值$P_{i,j}$与阈值$T_{i，j}$之间的差值作为sigmod函数的输出，然后再经过放大系数$k$，将其输出无限逼近两个极端 0 或者1；

![](http://image.haibin.online/2020-09-11,19:04:07.jpg)


我们来根据label generation中的gt 与 threshold_map来分别计算下。经过这个可微分二值化的sigmod函数后，各个区域的像素值会变成什么样子：


文字实例中心区域像素：
* probability map 的gt为 1
* threshold map的gt值为0.3

所以：

$$\begin{aligned}
    binary_{x,y}&=\frac{1}{1+e^{-k\left(1-0.3\right)}}\\
    &=\frac{1}{1+e^{-k*0.7}}
\end{aligned}$$

如果不经过放大系数K的放大，那么区域正中心的像素如上图所示经过sigmod函数后趋向于0.6左右的值。但是经过放大系数k后，会往右倾向于1。



文字实例边缘区域像素：
* probability map 的gt为 1
*  threshold map的gt值为0.7

$$\begin{aligned}
    binary_{x,y}&=\frac{1}{1+e^{-k\left(1-0.7\right)}}\\
    &=\frac{1}{1+e^{-k*0.3}}
\end{aligned}$$

如果不经过放大系数K的放大，那么区域正中心的像素如上图所示经过sigmod函数后趋向于0.5左右的值。但是经过放大系数k后，会往右倾向于1。


文字实例外的像素：
* probability map 的gt为 0
*  threshold map的gt值为0.3

$$\begin{aligned}
    binary_{x,y}&=\frac{1}{1+e^{-k\left(0-0.3\right)}}\\
    &=\frac{1}{1+e^{-k*-0.3}}
\end{aligned}$$

经过放大系数k后，激活值会无限趋近于0；
从而实现二值化效果。


解释了DB利用类似sigmod的函数是如何实现二值化的效果，那么我们来看其梯度的学习性：

传统二值化是一个分段函数，如下图所示：
![](http://image.haibin.online/2020-09-11,19:24:21.jpg)

SB：standard binarization其梯度在0值被截断无法进行有效地回传。
DB：differentiable binarization是一个可微分的曲线，可以利用训练数据+优化器的形式进行数据驱动的学习优化。

我们来看其导数公式，假设$l_{+}$代表了正样本，$l_{-}$代表了负样本，则：

$$\begin{aligned}{l}
l_{+}&=-\log \frac{1}{1+e^{-k x}} \\
l_{-}&=-\log \left(1-\frac{1}{1+e^{-k x}}\right)
\end{aligned}$$

根据链式法则我们可以计算其loss梯度：


### DB的pipeline


![](http://image.haibin.online/2020-09-11,16:47:33.jpg)


整体流程如图所示：
* backbone网络提取图像特征
* 类似FPN网络结构进行图像特征融合后得到两个特征图 probability map 跟 threshold map
* probability map 与threshold map 两个特征图做DB差分操作得到文字区域二分图
* 二分图经过cv2 轮廓得到文字区域信息



### 思考与讨论

（1） DB精髓是通过学习得到一个Adaptive threshold map，以更好地适应分割概率图中转变为文字像素二分图的准确率。为了这个threshold map 阈值图变得可学习，将传统的阈值分割转变为一个sigmod函数。
















