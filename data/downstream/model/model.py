import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pdb

sys.path.append('/data/private/BERT/kor-bert-base-dha_bpe.v3/')
from dha_rung import DHATokenizer_Rung

from transformers import BertTokenizer, BertModel
from transformers import BertConfig

class EngModel(nn.Module):
    def __init__(self, cls_num, scratch):
        super().__init__()
        bert_path = '/data/private/BERT/bert_base_uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        self.cls_num = cls_num # 243
        
        if scratch:
            config = BertConfig.from_pretrained(bert_path+'/config.json')
            self.model = BertModel(config)            
        else: # pretrained model
            self.model = BertModel.from_pretrained(bert_path)
        
        # classifier
        self.fc = nn.Linear(768, self.cls_num)
        
        """비학습 함수"""
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_tensor):
        """
        input_tensor: (1, token_len)
        """
        output = self.model(input_tensor).last_hidden_state
        output = output[:,0,:] # (len, 768)
        logit = self.fc(output) # (len, ner_num)
        
        return logit
    
    def clsLoss(self, out_logit, label):
        """
        out_logit: (batch, cls_num)
        label: (cls_num)
        """
        loss_val = self.loss(out_logit, label)
        return loss_val    
    
class KorModel(nn.Module):
    def __init__(self, cls_num, scratch):
        super().__init__()
        bert_path = '/data/private/BERT/kor-bert-base-dha_bpe.v3'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.cls_num = cls_num # 243        
        
        if scratch:
            config = BertConfig.from_pretrained(bert_path+'/config.json')
            self.model = BertModel(config)            
        else: # pretrained model
            self.model = BertModel.from_pretrained(bert_path)
        
        # classifier
        self.fc = nn.Linear(768, self.cls_num)
        
        """비학습 함수"""
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_tensor):
        """
        input_tensor: (1, token_len)
        """
        output = self.model(input_tensor).last_hidden_state
        output = output[:,0,:] # (len, 768)
        logit = self.fc(output) # (len, ner_num)
        
        return logit
    
    def clsLoss(self, out_logit, label):
        """
        out_logit: (batch, cls_num)
        label: (cls_num)
        """
        loss_val = self.loss(out_logit, label)
        return loss_val    