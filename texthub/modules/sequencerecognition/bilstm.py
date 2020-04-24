import torch.nn as nn
from ..registry import SEQUENCERECOGNITIONS
from ..utils.moduleinit import normal_init

@SEQUENCERECOGNITIONS.register_module
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def init_weights(self, pretrained=None):
        if pretrained==None:
            self.rnn.reset_parameters()
            normal_init(self.linear, std=0.01)
        elif isinstance(pretrained,str):
            #TODO:load pretrain
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
