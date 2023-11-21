import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import transformers
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, TrainingArguments, AdamW
import os
from tqdm import tqdm
import wandb
from dataset import PositiveDataset

from config import config


def main(config):

    # tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # not needed for unconstrained
    # tokenizer.add_special_tokens({'additional_special_tokens': [config['strategy_token', 'ref_token']]})


    # model
    # TODO: change model to bart-large
    if config['pretrained_path'] is None:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to('cuda')
    else:
        model = BartForConditionalGeneration.from_pretrained(config['pretrained_path']).to("cuda")
    
    model.train()
    
    # dataset
    dataset = PositiveDataset("/workspace/positive_reframe/data", phase='train', tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'])
    
    # optimizer
    optim = AdamW(model.parameters(), lr=config['init_lr'])

    # wandb logging
    if config['wandb']:
        wandb.init(
            name=config['run_name'],
            project='pos_reframing',
            entity='hjjung',
            reinit=True,
            tags=['bart'],
            save_code=True
        )

    # train
    for epoch in range(config['epochs']):
        total_loss = 0
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
        for i, batch in tqdm(enumerate(dataloader)):
            optim.zero_grad()

            input_text = batch['original_text']
            label_text = batch['reframed_text']
            strategy = batch['strategy']

            # unconstrained
            input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128,truncation=True, padding='max_length')
            input_ids = input_tokens['input_ids'].to('cuda')
            attention_mask = input_tokens['attention_mask'].to('cuda')

            labels = tokenizer(label_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')['input_ids'][:,1:].to('cuda')
            labels[labels==tokenizer.pad_token_id] = -100

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss

            batch_bar.set_postfix(
                loss = "{:.04f}".format(float(total_loss / (i+1))),
                lr = "{:.04f}".format(float(optim.param_groups[0]['lr']))
            )

            if config['wandb']:
                wandb.log({"train_loss": total_loss / (i+1)})

            loss.backward()
            optim.step()
            batch_bar.update()
            model.save_pretrained(f"{config['save_path']}/{config['run_name']}")

        batch_bar.close()

if __name__=='__main__':
    main(config)