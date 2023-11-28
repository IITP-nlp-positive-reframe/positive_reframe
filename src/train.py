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
from datasets import load_metric

from config import config


def main(config):

    # tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # not needed for unconstrained
    # tokenizer.add_special_tokens({'additional_special_tokens': [config['strategy_token', 'ref_token']]})


    # model
    # TODO: change model to bart-large
    if config['pretrained_path'] is None:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(config['device'])
    else:
        model = BartForConditionalGeneration.from_pretrained(config['pretrained_path']).to(config['device'])
    
    model.train()
    
    # dataset
    dataset = PositiveDataset("/workspace/positive_reframe/data", phase='train', tokenizer=tokenizer)
    val_dataset = PositiveDataset("/workspace/positive_reframe/data", phase='dev', tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # optimizer
    optim = AdamW(model.parameters(), lr=config['init_lr'])

    # wandb logging
    if config['wandb']:
        wandb.login(key="a4685c36eca1cc5a3eb4745aab432b225d0ba421")
        wandb.init(
            name=config['run_name'],
            project='pos_reframing',
            entity='hjjung',
            reinit=True,
            tags=['bart'],
            save_code=True
        )

    # eval metric
    bleu = load_metric("sacrebleu")
    best_val_score = 0

    # train
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
        for i, batch in tqdm(enumerate(dataloader)):
            optim.zero_grad()

            input_text = batch['original_text']
            label_text = batch['reframed_text']
            strategy = batch['strategy']

            # unconstrained
            input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128,truncation=True, padding='max_length')
            input_ids = input_tokens['input_ids'].to(config['device'])
            attention_mask = input_tokens['attention_mask'].to(config['device'])

            labels = tokenizer(label_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')['input_ids'].to(config['device'])
            # [:,1:].to(config['device'])
            labels[labels==tokenizer.pad_token_id] = -100

            # import pdb; pdb.set_trace()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.detach().float()

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

        # eval
        if epoch % 5 == 0 or epoch==config['epochs']-1:
            model.eval()
            gts = []
            preds = []
            with torch.no_grad():
                for i, val_batch in enumerate(val_loader):
                    input_text = val_batch['original_text']
                    label_text = val_batch['reframed_text']

                    input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128,truncation=True, padding='max_length')
                    input_ids = input_tokens['input_ids'].to(config['device'])

                    output_ids = model.generate(input_ids, num_beams=config['num_beams'], min_length=0, max_length=128)
                    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    preds.append([output_text])
                    gts.append([label_text[0]]) # ?? evaluate.py에 요상하게 해둠. 왜지
                bleu_scores = bleu.compute(predictions=preds, references=gts)['score']
                if config['wandb']:
                    wandb.log({'bleu score': bleu_scores})
                print("bleu score: ", bleu_scores)
                if bleu_scores > best_val_score:
                    best_val_score = bleu_scores
                    model.save_pretrained(f"{config['save_path']}/{config['run_name']}/best")
                if epoch==config['epochs']-1:
                    model.save_pretrained(f"{config['save_path']}/{config['run_name']}/last")



if __name__=='__main__':
    main(config)
