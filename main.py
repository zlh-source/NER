from util import *
from transformers import BertTokenizer
import argparse
import json
from tqdm import tqdm
from NERModel import NERModel
import os
import torch
import torch.nn.functional as F
import pickle

class data_generator(DataGenerator):
    def __init__(self, args, data, id2tag, tag2id,cache_path, random=True): #原始数据  转化为 模型接受的矩阵
        super(data_generator, self).__init__(data, args.batch_size)
        self.random = random
        self.max_len = 512
        self.id2tag=id2tag
        self.tag2id=tag2id
        self.all_ex=data

        if not os.path.exists(os.path.join(args.dataset_path, "cache")):
            os.makedirs(os.path.join(args.dataset_path, "cache"))
        if not os.path.exists(cache_path):
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=True)  # 添加特殊字符
            self.all_token_id = np.ones([len(data), self.max_len], dtype=np.int64) * self.tokenizer._convert_token_to_id('[PAD]') #512
            self.all_mask = np.zeros([len(data), self.max_len], dtype=np.int64)
            self.all_label = np.ones([len(data), self.max_len], dtype=np.int64)*tag2id["O"]
            for ex_index,ex in tqdm(enumerate(data),total=len(data),desc='data_loader'):
                text=ex[0]
                token_id = get_token_id(self.tokenizer,text) #52
                for start,end,c in ex[1:]:
                    #对序列中的 start+1,end+1进行标注
                    self.all_label[ex_index][start+1]=tag2id["B-"+c]
                    self.all_label[ex_index][start + 2:end+2] = tag2id["I-" + c]
                self.all_token_id[ex_index,:len(token_id)]=token_id
                self.all_mask[ex_index][:len(token_id)]=1
            print("构建%s" % (cache_path))
            cache_dict = dict(zip(['token_id', 'mask','label'],[self.all_token_id, self.all_mask,self.all_label]))
            pickle.dump(cache_dict, open(cache_path, 'wb'), protocol=4)
        else:
            print("从%s读取数据"%(cache_path))
            cache_dict = pickle.load(open(cache_path, 'rb'))
            self.all_token_id = cache_dict["token_id"]
            self.all_mask = cache_dict["mask"]
            self.all_label = cache_dict["label"]

    def __iter__(self):
        batch_index = []
        for is_end, index in self.sample(self.random):
            batch_index.append(index)
            if len(batch_index) == self.batch_size or is_end: #是否凑够一个batch  或者 是否读完所有数据
                bsz=len(batch_index)
                seq_len=self.all_mask[batch_index].sum(axis=-1).max() #一个batch中样本的最大句子长度
                batch_token_id=self.all_token_id[batch_index][:,:seq_len]
                batch_mask = self.all_mask[batch_index][:, :seq_len]
                batch_label = self.all_label[batch_index][:, :seq_len]
                batch_ex=[self.all_ex[ex_index] for ex_index in batch_index]
                yield [
                    batch_token_id,batch_mask,batch_label,batch_ex
                    ]
                batch_index = []

class test_data_generator(DataGenerator):
    def __init__(self, args, data, id2tag, tag2id,cache_path, random=False):
        super(test_data_generator, self).__init__(data, args.batch_size)
        self.random = random
        self.max_len = 512
        self.id2tag=id2tag
        self.tag2id=tag2id
        self.all_ex=data

        if not os.path.exists(os.path.join(args.dataset_path, "cache")):
            os.makedirs(os.path.join(args.dataset_path, "cache"))
        if not os.path.exists(cache_path):
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=True)  # 添加特殊字符
            self.all_token_id = np.ones([len(data), self.max_len], dtype=np.int64) * self.tokenizer._convert_token_to_id('[PAD]')
            self.all_mask = np.zeros([len(data), self.max_len], dtype=np.int64)
            self.all_label = np.ones([len(data), self.max_len], dtype=np.int64)*tag2id["O"]
            for ex_index,ex in tqdm(enumerate(data),total=len(data),desc='data_loader'):
                text=ex["text"]
                token_id = get_token_id(self.tokenizer,text)
                self.all_token_id[ex_index,:len(token_id)]=token_id
                self.all_mask[ex_index][:len(token_id)]=1
            print("构建%s" % (cache_path))
            cache_dict = dict(zip(['token_id', 'mask'],[self.all_token_id, self.all_mask]))
            pickle.dump(cache_dict, open(cache_path, 'wb'), protocol=4)
        else:
            print("从%s读取数据"%(cache_path))
            cache_dict = pickle.load(open(cache_path, 'rb'))
            self.all_token_id = cache_dict["token_id"]
            self.all_mask = cache_dict["mask"]

    def __iter__(self):
        batch_index = []
        for is_end, index in self.sample(self.random):
            batch_index.append(index)
            if len(batch_index) == self.batch_size or is_end:
                bsz=len(batch_index)
                seq_len=self.all_mask[batch_index].sum(axis=-1).max()
                batch_token_id=self.all_token_id[batch_index][:,:seq_len]
                batch_mask = self.all_mask[batch_index][:, :seq_len]
                batch_ex=[self.all_ex[ex_index] for ex_index in batch_index]
                yield [
                    batch_token_id,batch_mask,batch_ex
                    ]
                batch_index = []

def train(args):
    output_path=os.path.join(args.dataset_path, args.file_id) #./dataset/OIE/openie6_data/file_id
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print_config(args)  # 输出超参数
    train_data, categories = load_data(os.path.join(args.dataset_path,"train.json"))
    dev_data,_ = load_data(os.path.join(args.dataset_path,"dev.json"))

    categories = sorted(categories) #排序后的实体类别列表
    id2tag, tag2id=get_categories_map(categories)

    train_dataloader=data_generator(args, train_data,id2tag, tag2id,cache_path=os.path.join(args.dataset_path,"cache","train"))
    dev_dataloader=data_generator(args, dev_data,id2tag, tag2id,cache_path=os.path.join(args.dataset_path,"cache","dev"),random=False)

    train_model=NERModel(args,len(tag2id))
    train_model.to(args.device)
    optimizer,scheduler=get_optimizer(len(train_dataloader),args,train_model)
    log_path=os.path.join(output_path, 'log.txt')

    best_f1=0.0
    for epoch in range(args.num_train_epochs):
        train_model.train()
        step_loss = 0.0
        with tqdm(total=len(train_dataloader),desc="epoch %d"%epoch) as t:

            for i,d in enumerate(train_dataloader):
                batch_token_id,batch_mask,batch_label = \
                    [torch.tensor(i).to(args.device) for i in d[:-1]]
                logits=train_model(batch_token_id,batch_mask) #B L R
                loss=F.cross_entropy(input=logits.flatten(0,1),target=batch_label.flatten(0,1),reduction="sum")
                loss.backward()
                step_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                train_model.zero_grad()
                t.set_postfix({'loss' : loss.item()})
                t.update(1)

        #torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))
        f1, pre, rec = test(args, dev_dataloader, train_model,id2tag)
        if f1>best_f1: #出现更好的分数， 存储权重，更新f1
            best_f1=f1
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))
        with open(log_path,"a") as f:
            print("epoch:%d\tpre=%f\trec=%f\tf1=%f\tbest_f1=%f"%(epoch,pre, rec, f1,best_f1),file=f)
        print("epoch:%d\tpre=%f\trec=%f\tf1=%f\tbest_f1=%f"%(epoch,pre, rec, f1,best_f1))


def test(args,dataloader=None,model=None,id2tag=None):
    print('测试')
    output_path = os.path.join(args.dataset_path, args.file_id)
    if not dataloader or not model or not id2tag:
        dev_data, categories = load_data(os.path.join(args.dataset_path, 'dev.json'))
        categories = sorted(categories)
        id2tag, tag2id = get_categories_map(categories)
        dataloader = data_generator(args, dev_data, id2tag, tag2id,cache_path=os.path.join(args.dataset_path,"cache","dev"),random=False)
        model = NERModel(args,len(tag2id))
        model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME)))
    model.eval()
    X, Y, Z = 1e-10, 1e-10, 1e-10
    all_pred=[]
    with torch.no_grad():
        for d in tqdm(dataloader,total=len(dataloader),desc='dev'):
            batch_ex=d[-1]
            batch_token_id, batch_mask, _ = \
                [torch.tensor(i).to(args.device) for i in d[:-1]]
            batch_logits=model(batch_token_id, batch_mask).detach().cpu().numpy() #bsz seq_lens num_tags
            for ex_index,ex in enumerate(batch_ex):
                T=set(ex[1:])
                tag_seq = batch_logits[ex_index].argmax(axis=-1).tolist()
                R=get_entity_from_tag_seq(tag_seq, id2tag, ex[0])
                X += len(R & T) # 预测正确的实体数量
                Y += len(R) #
                Z += len(T)
                all_pred.append({"text":ex[0],"pred":list(R),"target":list(T),'new': list(R - T),'lack': list(T - R)})
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z #以实体为粒度计算混淆矩阵(F1)

    json.dump(all_pred,open(os.path.join(output_path,"dev_pred.json"),"w",encoding='utf-8'),ensure_ascii=False,indent=4)
    print("f1:%f, precision:%f, recall:%f"%(f1, precision, recall))
    return f1, precision, recall

def get_predict(args,dataloader=None,model=None,id2tag=None):
    print('预测')
    output_path = os.path.join(args.dataset_path, args.file_id)
    if not dataloader or not model or not id2tag:
        _, categories = load_data(os.path.join(args.dataset_path, "train.json"))
        categories = sorted(categories)
        id2tag, tag2id = get_categories_map(categories)
        test_data=[json.loads(l) for l in open(os.path.join(args.dataset_path, "test.json"))]
        dataloader = test_data_generator(args, test_data, id2tag, tag2id,cache_path=os.path.join(args.dataset_path,"cache","test"),random=False)
        model = NERModel(args,len(tag2id))
        model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME)))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    all_pred=[]
    with torch.no_grad():
        for d in tqdm(dataloader,total=len(dataloader),desc='test'):
            batch_ex=d[-1]
            batch_token_id, batch_mask = \
                [torch.tensor(i).to(args.device) for i in d[:-1]]
            batch_logits=model(batch_token_id, batch_mask).detach().cpu().numpy()
            for ex_index,ex in enumerate(batch_ex):
                tag_seq = batch_logits[ex_index].argmax(axis=-1).tolist()
                R=get_entity_from_tag_seq(tag_seq, id2tag, ex["text"])
                item={"id":ex["id"],"label":{}}
                for start,end,c in list(R):
                    entity=ex["text"][start:end+1]
                    if c not in item["label"]:
                        item["label"][c]={}
                    if entity not in item["label"][c]:
                        item["label"][c][entity]=[]
                    item["label"][c][entity].append([start,end])
                all_pred.append(item) #{"id": 0, "label": {"address": {"丹棱县": [[11, 13]]}, "name": {"胡文和": [[41, 43]]}}}
    with open(os.path.join(output_path,"cluener_predict.json"),"w",encoding="utf-8") as f:
        for ex in all_pred:
            ex = json.dumps(ex, ensure_ascii=False)
            f.write(ex + '\n')



