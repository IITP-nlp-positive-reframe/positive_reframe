import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import wandb
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, TrainingArguments, AdamW
from datasets import load_metric
from dataset import PositiveDataset
from config import config
import nltk
from textblob import TextBlob
import numpy as np
import evaluate

# tokenizer
# TODO: adjust tokenizer path to the same as model (bart-large/base...)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
tokenizer.add_special_tokens({'additional_special_tokens': [config['strategy_token'], config['ref_token']]})

# model
# TODO: change to bart-large
model = BartForConditionalGeneration.from_pretrained(config['pretrained_path']).to(config['device'])
model.eval()

# dataset
dataset = PositiveDataset("/workspace/positive_reframe/data", phase='test', tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1) # test batch size=1

bleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

inputs = []
gts = []
preds = []
eval_file_path = f"{config['pretrained_path']}/test_eval.txt"
with open(eval_file_path, 'w') as f:
    f.close()
with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader)):
        input_text = batch['original_text']
        label_text = batch['reframed_text']
        strategy = batch['strategy']

        input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        input_ids = input_tokens['input_ids'].to(config['device'])

        output_ids = model.generate(input_ids, num_beams=config['num_beams'], min_length=0, max_length=128)
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        inputs.extend(input_text)
        preds.extend(output_text)
        gts.append(label_text)
        with open(eval_file_path, 'a') as f:
            for idx in range(len(output_text)):
                f.write("input: "+input_text[idx] + "\npred: " + output_text[idx] + "\ngt: " + label_text[idx] + "\n\n")



"""
# preds: ["sentence1", "sentence2", ...]
# gts: [["sentence1"], ["sentence2"], ...] 
#   -> bleu에 gt로 들어가는 건 이중 list 형태여야 함
#   -> bertscore는 두가지 형태 다 가능이라고 함
#   -> rouge 계산할 때는 flatten해야 함
# inputs: ["sentence1", "sentence2", ...]
"""

### ROUGE
pred_for_rouge = ['\n'.join(nltk.sent_tokenize(p.strip())) for p in preds]
gt_for_rouge = ['\n'.join(nltk.sent_tokenize(g[0].strip())) for g in gts]
# print("sanity check")
# with open(eval_file_path.split(".")[0] + "_rouge.txt", 'w') as f:
#     f.write(pred_for_rouge + "\n\n\n" + gt_for_rouge)
# print(pred_for_rouge)
# print(gt_for_rouge)
rouge_scores = rouge.compute(predictions=pred_for_rouge, references=gt_for_rouge, use_stemmer=True)
print("load_metric('rouge1'):", rouge_scores['rouge1'])
print("load_metric('rouge2'):", rouge_scores['rouge2'])
print("load_metric('rougeL'):", rouge_scores['rougeL'])
print("load_metric('rougeLsum'):", rouge_scores['rougeLsum'])
with open(eval_file_path, 'a') as f:
    f.write("rouge: " + str(rouge_scores))

### BLEU
bleu_scores = bleu.compute(predictions=preds, references=gts)['score']
print("load_metric('sacrebleu')", bleu_scores)
with open(eval_file_path, 'a') as f:
    f.write("\nbleu: " + str(bleu_scores))

### BERTSCORE
bert_scores = bertscore.compute(predictions=preds, references=gts, lang='en')['f1']
bert_scores = np.mean(bert_scores)
print("load_metric('bertscore')", bert_scores)
with open(eval_file_path, 'a') as f:
    f.write("\nbertscore: " + str(bert_scores))

### TEXTBLOB SENTITMENT POLARITY DELTA
pred_sentiment_scores = []
input_sentiment_scores = []
assert len(preds) == len(gts)
for i in range(len(preds)):
    blob = TextBlob(preds[i])
    pred_sentiment_scores.append(blob.sentences[0].sentiment.polarity)
    blob = TextBlob(inputs[i])
    input_sentiment_scores.append(blob.sentences[0].sentiment.polarity)
pred_sent_scores = np.array(pred_sentiment_scores)
input_sent_scores = np.array(input_sentiment_scores)
deltas = pred_sent_scores - input_sent_scores
avg_delta = deltas.mean()
print("Text Blob Avg Sentiment Change: ", avg_delta)
with open(eval_file_path, 'a') as f:
    f.write("\ntextblob sent score change: " + str(avg_delta))
