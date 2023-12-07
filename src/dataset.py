import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

    
class PositiveDataset(Dataset):
    def __init__(self, root, phase, tokenizer):
        # exp_setting='unconstrained'
        # , condition='original', output_type='reframed'
        self.root = root
        self.phase = phase
        self.tokenizer = tokenizer
        # self.exp_setting = exp_setting
        # self.condition = condition
        # self.output_type = output_type

        assert self.phase in ['train', 'dev', 'test']
        # assert self.exp_setting in ['unconstrained', 'controlled', 'predict']
        # assert self.condition in ['original', 'strategy']
        # assert self.output_type in ['reframed', 'strategy']

        # [unconstrained] condition: original - output_type: reframed 
        # [controlled] condition: strategy - output_type: reframed 
        # [predict] condition: original - output_type: strategy 
        

        self.csv = pd.read_csv(os.path.join(self.root, f'{self.phase}.csv'))
        self.csv = dict(self.csv) # keys: original_text, reframed_text, strategy, original_with_label
        for key in self.csv:
            self.csv[key] = list(self.csv[key])
            
    def __len__(self):
        return len(self.csv['original_text'])
    
    def __getitem__(self, idx):
        original_text = self.csv['original_text'][idx]
        reframed_text = self.csv['reframed_text'][idx]
        strategy = self.csv['strategy'][idx]

        output = {
            "original_text": original_text,
            "reframed_text": reframed_text,
            "strategy": strategy
        }

        return output
    