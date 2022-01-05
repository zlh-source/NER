import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer,BertForSequenceClassification

class NERModel(nn.Module):
    def __init__(self,args, num_tags):
        super(NERModel, self).__init__()
        self.num_tags=num_tags
        self.bert = BertModel.from_pretrained(args.bert_directory)
        self.linear=nn.Linear(self.bert.config.hidden_size,num_tags)
    def forward(self,token_ids,mask_ids):
        '''
        :param token_ids: [bsz,seq_len]
        :param mask_ids: [bsz,seq_len]
        :return:
        '''
        h = self.bert(input_ids=token_ids, attention_mask=mask_ids)[0]  #[bsz,seq_len,hidden_size]
        logit=self.linear(h) #[bsz,seq_len,num_tags]
        return logit