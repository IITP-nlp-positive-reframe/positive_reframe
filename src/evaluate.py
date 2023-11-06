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


# tokenizer
# TODO: adjust tokenizer path to the same as model (bart-large/base...)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_special_tokens({'additional_special_tokens': [config['strategy_token'], config['ref_token']]})

# model
# TODO: change to bart-large
model = BartForConditionalGeneration.from_pretrained(config['pretrained_path']).to("cuda")
model.eval()

# dataset
dataset = PositiveDataset("../data", phase='test', tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1) # test batch size=1

bleu = load_metric('sacrebleu')

gts = []
preds = []

for i, batch in tqdm(enumerate(dataloader)):
    input_text = batch['original_text']
    label_text = batch['reframed_text']
    strategy = batch['strategy']

    input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    input_ids = input_tokens['input_ids'].to('cuda')

    output_ids = model.generate(input_ids, num_beams=config['num_beams'], min_length=0, max_length=128)
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    preds.append([output_text])
    gts.append(label_text if len(label_text)==1 else label_text)

bleu_scores = bleu.compute(predictions=preds, references=gts)['score']
print("load_metric('sacrebleu')", bleu_scores)
