import argparse
import random
import numpy as np
import torch
import os

def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

from main import *

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--cuda_id', default="7", type=str,help='显卡号')
parser.add_argument('--cpu', default=True, type=bool,help='是否使用cpu?')

parser.add_argument('--dataset_path', default='./datasets/CLUENER', type=str, help="")
parser.add_argument('--train', default="train", type=str,help="train or test or predict")
parser.add_argument('--file_id', default="00", type=str)

#超参数
#Encoder

#Decoder

#loss
parser.add_argument('--info', default='', type=str,help='关于模型的描述')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=3e-5, type=float)
parser.add_argument('--num_train_epochs', default=10, type=int)

#基本不用改
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)

#不用改
parser.add_argument('--bert_directory', default="./pretrained/base-chinese", type=str)

args = parser.parse_args()


print('begin')
if args.cpu:
    device = "cpu"
else:
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = "cuda:"+args.cuda_id

args.device = torch.device(device)

if args.train=="train":
    train(args)
elif args.train=="test":
    test(args)
else:
    get_predict(args)