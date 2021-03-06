
import string
import numpy as np
import os
this_path = os.path.split(os.path.realpath(__file__))[0]

def getChineseCharset():
    with open(os.path.join(this_path,'./resources/new_total_chinese_charset.dic'),'r') as file_handler:
        corups = file_handler.read().rstrip().strip()
    # with open(os.path.join(this_path,'./resources/chinese_charset.dic'),'r') as file_handler:
    #     corups = file_handler.read().rstrip().strip()
    return corups


ChineseCharset = getChineseCharset()
EnglishSensitiveCharset = string.digits + string.ascii_letters + string.punctuation
EnglishNoSensitiveCharset = "0123456789abcdefghijklmnopqrstuvwxyz"
CharsetDict = {
    "ChineseCharset":ChineseCharset,
    "EnglishSensitiveCharset":EnglishSensitiveCharset,
    "EnglishNoSensitiveCharset":EnglishNoSensitiveCharset
}



