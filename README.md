## Chinese NER

## Model
Bert+FFN

## Requirements
The main requirements are:
- python 3.6
- torch 1.7.0 
- tqdm
- transformers 3.5.1

## Dataset
Download [CLUNER](https://www.cluebenchmarks.com/introduce.html) and put it under `./datasets/CLUENER`.

## Usage
* **Get pre-trained BERT model**
Download [bert-base-chinese](https://huggingface.co/bert-base-chinese) and put it under `./pretrained`.

* **Train**
```
python run.py --train train
```

* **Evaluate on the Test**
```
python run.py --train predict
```

## Result(F1)
|  dev   | test  |
|  :----:  | :----:  |
|  76.2  |  77.5  |
