import torch
import torch.nn as nn

from ..registry import  HEADS
from .labelconverter import CTCLabelConverter
from ...datasets.pipelines import CTCChineseCharsetConverter

@HEADS.register_module
class CTC2DHead(nn.Module):
    def __init__(self,input_size,charsets,batch_max_length=25,inner_channels=256, stride=1):
        super(CTC2DHead, self).__init__()

        self.converter = CTCChineseCharsetConverter(charsets,batch_max_length)
        self.num_class = len(self.converter)
        self.batch_max_length = batch_max_length

        self.inner_channels = inner_channels
        self.pred_mask_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=(stride,stride),
                         stride=(stride,stride)),
            nn.Conv2d(in_channels=input_size,out_channels=inner_channels,kernel_size=3,padding=1),
            nn.Conv2d(inner_channels,1,kernel_size=1),
            ##pred_mask为论文中的path transform map，[h,w,h'],从某一个h跳转到一层h’的概率和为1，所以softmax
            nn.Softmax(dim=2)
        )

        self.pred_classify_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=(stride,stride),
                         stride=(stride,stride)),
            nn.Conv2d(in_channels=input_size,out_channels=inner_channels,kernel_size=3,padding=1),
            nn.Conv2d(inner_channels,self.num_class,kernel_size=1)
        )

        self.tiny = torch.tensor(torch.finfo().tiny, requires_grad=False)
        self.register_buffer('saved_tiny', self.tiny)
        from ...ops.ctc_2d import ctc_loss_2d
        self.ctc_loss = ctc_loss_2d

        ##pytroch ctc2d loss
        # self.ctc_loss = CTCLoss2D()

    def init_weights(self,pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1e-4)



    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)




    def forward_train(self,data:dict):
        feature = data.get('img')
        text = data["label"]
        lengths = data["length"]


        masking = self.pred_mask_layer(feature)
        mask = masking
        classify = self.pred_classify_layer(feature)
        classify = nn.functional.softmax(classify, dim=1)


        # #use rec_ctc_loss2d  CTCLoss2D
        # #mask [batch,1,h,w] ->[w,h,batch]
        # path_trans_map = mask.squeeze(1).permute(2,1,0)
        # ##[batch,class,h,w]-> W, H, N, C
        # classify = classify.permute(3,2,0,1).contiguous()
        # classify = classify.log_softmax(3)
        # input_lengths = torch.full((classify.size(2),), classify.size(0), dtype=torch.long).to(classify.device)
        # path_trans_map = path_trans_map.log()
        # # print(path_trans_map.shape,classify.shape,torch.max(text),torch.max(lengths),input_lengths.shape)
        # loss_value = self.ctc_loss(mask=path_trans_map, classify=classify, targets=text, input_lengths=input_lengths, target_lengths=lengths)/ lengths.float()
        # return dict(
        #      loss=loss_value, ctc_2d_loss=loss_value
        # )






        # # use ops.ctc2d loss
        tiny = self.saved_tiny
        # print(mask.shape,classify.shape)
        ##mask 为path transition map   mask 与classify 相乘得到 2-d 概率图的attention后的概率图
        pred = mask * classify # N, C, H ,W
        ##log 为了计算ctc loss准备
        pred = torch.log(torch.max(pred, tiny))
        pred = pred.permute(3, 2, 0, 1).contiguous()  # W, H, N, C
        input_lengths = torch.zeros(
            (feature.size()[0],)) + pred.shape[0]

        # lengths = lengths.squeeze(1)
        loss = self.ctc_loss(pred, text.long(), input_lengths.long().to(
            pred.device), lengths.long()) / lengths.float()
        """
        // -ln（p）=0即证明，概率为1，所以不能单纯为0，而是使得概率P是一个特别小的数字
            //-ln(0.001)=6.9
            //两个概率都为0 log（-inf）=0.0，
        """
        no_inf_loss = torch.where(torch.isinf(loss),torch.full_like(loss,6.9),loss)
        ##避免因为有个inf 导致loss.mean为inf
        return dict(
            loss=no_inf_loss, ctc_2d_loss=no_inf_loss
        )

    def forward_test(self,data:dict):
        feature = data.get('img')
        # tiny = self.saved_tiny
        masking = self.pred_mask_layer(feature)
        #（N，1，H，W）
        mask = masking
        classify = self.pred_classify_layer(feature)
        #（N，C,H，W）
        classify = nn.functional.softmax(classify, dim=1)
        return mask,classify

    def postprocess(self, mask: torch.Tensor,classify:torch.Tensor):

        heatmap = classify * mask
        # classify = classify.to('cpu')
        # mask = mask.to('cpu')
        paths = heatmap.max(1, keepdim=True)[0].argmax(
            2, keepdim=True)  # (N, 1, 1, W)
        C = classify.size(1)
        paths = paths.repeat(1, C, 1, 1)  # (N, C, 1, W)
        selected_probabilities = heatmap.gather(2, paths)  # (N, C, W)

        pred = selected_probabilities.argmax(1).squeeze(1)  # (N, W)

        output = torch.zeros(
                pred.shape[0], pred.shape[-1], dtype=torch.int) + self.converter.blank
        pred = pred.to('cpu')
        output = output.to('cpu')

        for i in range(pred.shape[0]):
            valid = 0
            previous = self.converter.blank
            for j in range(pred.shape[1]):
                c = pred[i][j]
                if c == previous or c == self.converter.unknown:
                    continue
                if not c == self.converter.blank:
                    output[i][valid] = c
                    valid += 1
                previous = c
        pred_strings = [self.converter.label_to_string(pred) for pred in output ]
        scores = [0 for _ in output]
        return pred_strings,scores
        # preds_size = torch.IntTensor([self.batch_max_length] * mask.size(0))
        #
        # preds_strs = self.converter.decode(preds_index, preds_size)
        # scores = []
        # return preds_strs,scores




