import json
import numpy as np
import os
from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup

def get_entity_from_tag_seq(tag_seq,id2tag,text):
    '''
    :param tag_seq:
    :param id2tag:
    :param text:
    :return: R=set( (start, end, c) )
    '''
    num_entity = 0
    R=set()
    start, end, c = -1, -1, None
    for token_index, tag_id in enumerate(tag_seq):
        if token_index == 0:
            continue
        if token_index > len(text):
            break
        tag = id2tag[tag_id]
        if tag[0] == 'B':  # 统计实体个数，做校验
            num_entity += 1
        if start == -1:  # 没有正在获取的实体
            if tag[0] == 'B':  # 有新实体
                start, end, c = token_index, token_index, tag[2:]
        else:  # 有正在获取的实体
            if tag[0] == 'O' or tag != 'I-' + c:  # 该实体结束
                R.add((start - 1, end - 1, c))
                if tag[0] in ['O', 'I']:  # 没有新实体
                    start, end, c = -1, -1, None
                else:  # 有新实体
                    start, end, c = token_index, token_index, tag[2:]
            else:  # 实体仍在继续获取
                end = token_index
    if start != -1:
        R.add((start - 1, end - 1, c))
    assert num_entity == len(R)
    return R


def get_optimizer(data_size,args,train_model):
    t_total = data_size * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )
    return optimizer,scheduler

def print_config(args):
    config_path=os.path.join(args.dataset_path, args.file_id,"config.txt")
    with open(config_path,"w",encoding="utf-8") as f:
        for k,v in sorted(vars(args).items()):
            print(k,'=',v,file=f)

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    categories=set() #所有的实体类别集合
    D = [] #所有的数据
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l) #str -> dic
            d = [l['text']]
            for k, v in l['label'].items(): #把字段转化为 [(k,v)]的形式  k:实体类型
                categories.add(k) #集合去重
                for spans in v.values(): #{'CSOL':[[4,7],[12,15]]}
                    for start, end in spans:
                        d.append((start, end, k)) #（实体开始位置，实体结束位置，实体类型）
            D.append(d)
    return D,categories

def get_categories_map(categories):
    id2tag, tag2id = {0: "O"}, {"O": 0}
    for c in categories: #c 实体类别
        id2tag[len(id2tag)] = 'B-%s' % (c)
        id2tag[len(id2tag)] = 'I-%s' % (c)
        tag2id['B-%s' % (c)] = len(tag2id)
        tag2id['I-%s' % (c)] = len(tag2id)
    return id2tag, tag2id

def get_token_id(tokenizer,text):
    token_id = [tokenizer._convert_token_to_id('[CLS]')] + \
               [tokenizer.encode(c)[1] for c in text] + \
               [tokenizer._convert_token_to_id('[SEP]')]
    return token_id

class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data=list(range(len(data)))
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d
