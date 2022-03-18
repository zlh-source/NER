import argparse
from main import train,test,get_predict
import torch
from util import seed_everything

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--cuda_id', default="0,2,3", type=str,help='显卡号')
parser.add_argument('--cpu', default=False, type=bool,help='是否使用cpu?')

parser.add_argument('--dataset_path', default='./datasets/CLUENER', type=str, help="")
parser.add_argument('--train', default="test", type=str,help="train or test or predict")
parser.add_argument('--file_id', default="99", type=str)

parser.add_argument('--seed', default=0, type=int)

#超参数
#Encoder

#Decoder

#loss
parser.add_argument('--info', default='', type=str,help='关于模型的描述')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=3e-5, type=float)
parser.add_argument('--num_train_epochs', default=10, type=int)

#基本不用改
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)

#不用改
parser.add_argument('--bert_directory', default="base-chinese", type=str)

args = parser.parse_args()


print('begin')
args.cuda_id=list(map(int,args.cuda_id.split(",")))

if args.cpu:
    device = "cpu"
else:
    device = "cuda:%d"%(args.cuda_id[0])

args.device = torch.device(device)

seed_everything(args.seed)

if args.train=="train":
    train(args)
elif args.train=="test":
    test(args)
else:
    get_predict(args)
