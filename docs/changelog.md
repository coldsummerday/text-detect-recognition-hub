



2020/07/29：

* 将detect结果增加为（batch_pred_bbox,score_bbox_list）
* 增加pan结果得分


2020/09/03:

* 修改了trainner，替换掉原来的runner模式（未完全测试
* 修改了logtexthook，输出更简洁
* 所有的hooks通过config的形式进行注册
* train.py统一入口，不再通过 texthub.api.train进行各种代码跳转
* 需要配合新的conifg文件进行测试。现只完成了pse的config，其他还没测试新的trainner

2020/09/04：
* 新增了warmupdecay的lr调整形式


2020/09/05：
* 发现了det_resnet在init_weights未启作用的bug。代码已修复，待测试


2020/09/05实验：
测试修改了backbone初始化后的pse，同时warmup lr调高至1e-5，warmup只在前5个epoch进行


2020/09/06:
db调整了epsilon的计算系数
一个连续光滑曲线折线化,epsilon阈值 距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越“接近”曲线。0.01的曲率过大
epsilon = 0.002 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

2020/09/16 实验
* 修改了pan的后处理pa计算，代码：texthub.ops.pa.src/pa.cpp
* 修改pan 的get_result
bug：pan 输出多边形的时候，eval时候用ploygon3 会出现segmentation fault，所以暂时先实现输出矩形框

2020/09/19 
* fix 了pan后处理的texthub.ops.pa.src/pa.cpp的数组下标 tempy,tempx越界问题
pan后处理输出矩形框、多边形 segmentation fault问题已经解决
* 合并了setup.py中编译cpp_ext的时候 include dir的问题，外部统一headers文件如pybind11放在texthub.ops.include中
避免多个拓展需要多次个include/pybind11文件
* 现pan,db已按照官方代码实现了一遍，暂时还没测试此版本代码在公共数据集如icdar上的表现（在小票数据已经进行测试
* pan ffem代码中上采样增加默认设置align_corners=False
原来：F.interpolate(x, size=y.size()[2:], mode='bilinear') + y
现在：F.interpolate(x, size=y.size()[2:], mode='bilinear',align_corners=False) + y
修改包括了：
c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear',align_corners=False)
c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear',align_corners=False)
c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear',align_corners=False)


