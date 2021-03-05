import torch
from ..registry import PIPELINES


@PIPELINES.register_module
class AttentionLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self, charsets,batch_max_length=25):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(charsets)
        self.character = list_token + list_character
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.batch_max_length = batch_max_length

    def encode(self,text,device):
        """ convert text-label into text-index.
        input:
            text: text label for image
        """
        length = len(text) # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length = self.batch_max_length + 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(batch_max_length + 1).fill_(0)
        text = list(text)
        text.append('[s]')
        text = [self.dict[char] for char in text]
        batch_text[1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token

        return batch_text.to(device),torch.IntTensor(length).to(device)

    def __call__(self, data:{}):
        img = data.get("img")
        label = data.get('label')
        if type(label)!=str:
            label = data.get("ori_label")
        device = img.device
        text,length_tensor= self.encode(label, device=device)
        data["ori_label"] = label
        data['label'] = text
        data["attn_text"]=text



        return data

@PIPELINES.register_module
class CTCLabelEncode(object):
    """ Convert between text-label and text-index """
    def __init__(self, charsets,batch_max_length=32):
        # character (str): set of the possible characters.
        dict_character = list(charsets)
        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        self.batch_max_length = batch_max_length

    def encode(self, text,device):
        """convert text-label into text-index.
        input:
            text: text labels of each image.

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = len(text)
        text = [self.dict[char] for char in text]
        if length > self.batch_max_length:
            length = self.batch_max_length
        batch_text = torch.LongTensor(self.batch_max_length).fill_(0)
        batch_text[:length] = torch.IntTensor(text[:length]).to(device)
        return (batch_text, torch.IntTensor([length]).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    def __call__(self, data:{}):
        img = data.get("img")
        label = data.get('label')
        if type(label)!=str:
            label = data.get("ori_label")
        device = img.device
        text_tensor,length_tensor= self.encode(label, device=device)
        data["ori_label"] = label
        data['label'] = text_tensor
        data["length"] = length_tensor
        data["ctc_text"] = text_tensor
        data["ctc_length"] = length_tensor
        return data



@PIPELINES.register_module
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, device,batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
