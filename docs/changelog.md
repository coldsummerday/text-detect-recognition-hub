



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
