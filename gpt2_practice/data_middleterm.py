import torch
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class Syllogistic_middleterm(Dataset):  
    def __init__(self, args, data, tokenizer, max_length, kfold_idx = None):
        self.data_path = os.path.join(args.data_path, data)
        self.raw_data = pd.read_csv(self.data_path)
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        
        self.tokenizer = tokenizer
        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token
        self.prem_connect = tokenizer.additional_special_tokens[0] # '<|and|>'
        self.conc = tokenizer.additional_special_tokens[1] #'<|so|>'
        self.input_pad_id = tokenizer.pad_token_id
        self.label_pad_id = -100
        
        self.prem1 = self.raw_data["1"].to_list() # premise 1
        self.prem2 = self.raw_data["3"].to_list() # premise 2
        self.not_middleterm1 = self.raw_data["5"].to_list() # not middleterm1
        self.not_middleterm2 = self.raw_data["7"].to_list() # not middleterm2
        
        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in enumerate(self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in enumerate(self.prem2) if num in kfold_idx]
            self.not_middleterm1 = [sent for num, sent in enumerate(self.not_middleterm1) if num in kfold_idx]
            self.not_middleterm2 = [sent for num, sent in enumerate(self.not_middleterm2) if num in kfold_idx]
            # self.label = [sent for num, sent in enumerate(self.label) if num in kfold_idx]

        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.not_middleterm1)== len(self.not_middleterm2),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n not_middletern1 : {len(self.not_middleterm1)} \n not_middletern2 : {len(self.not_middleterm2)}"

        
        self.input_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}" +not_middleterm1 + f"{self.prem_connect} " +not_middleterm2 + f"{self.eos}"  for p1, p2, not_middleterm1, not_middleterm2  in zip(self.prem1, self.prem2, self.not_middleterm1, self.not_middleterm2)]
        self.label_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}" +not_middleterm1 + f"{self.prem_connect} " +not_middleterm2 + f"{self.eos}"  for p1, p2, not_middleterm1, not_middleterm2  in zip(self.prem1, self.prem2, self.not_middleterm1, self.not_middleterm2)]
        self.input_for_generation_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"  for p1, p2 in zip(self.prem1, self.prem2)]
        self.conclusion_text = [label for label in self.label]
        self.conclusion_text = [not_mid1 + not_mid2  for not_mid1, not_mid2 in zip(self.not_middleterm1, self.not_middleterm2)]
    
    def __len__(self):
        return len(self.input_text)
        
    
    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx], self.label_text[idx], self.input_for_generation_text[idx], self.conclusion_text[idx])
    
    def __preprocess(self, input_text, label_text, input_for_generation_text, conclusion_text):
        input_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        label_token_ids = torch.full((1, self.max_len), fill_value = self.label_pad_id)
        input_for_generation_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        conclusion_token_ids = torch.full((1, self.max_len), fill_value = self.label_pad_id)

        attn_mask = torch.zeros((1, self.max_len))

        input_tokens = self.tokenizer.encode(input_text, add_special_tokens = True, return_tensors = 'pt')
        label_tokens = self.tokenizer.encode(label_text, add_special_tokens = True, return_tensors = 'pt')
        input_for_generation_tokens = self.tokenizer.encode(input_for_generation_text, add_special_tokens = True, return_tensors = 'pt')
        conclusion_tokens = self.tokenizer.encode(conclusion_text, add_special_tokens = True, return_tensors = 'pt')
        
        input_token_ids[0, :input_tokens.shape[1]] = input_tokens
        label_token_ids[0, :label_tokens.shape[1]] = label_tokens
        input_for_generation_token_ids[0, :input_for_generation_tokens.shape[1]] = input_for_generation_tokens
        conclusion_token_ids[0, :conclusion_tokens.shape[1]] = conclusion_tokens
        
        attn_mask[0, :input_tokens.shape[1]] = 1

        return input_token_ids, attn_mask, label_token_ids, input_for_generation_token_ids, conclusion_token_ids


def collate_fn(batch) :
    input_ids = []
    attn_mask = []
    label_ids = []
    input_for_generation_ids =[]
    conclusion_ids =[]
    
    for input_token_ids, attn_token_mask, label_token_ids, input_for_generation_token_ids, conclusion_token_ids in batch:
                
        input_ids.append(input_token_ids)
        attn_mask.append(attn_token_mask)
        label_ids.append(label_token_ids)
        input_for_generation_ids.append(input_for_generation_token_ids)
        conclusion_ids.append(conclusion_token_ids)
        
    model_inputs = {
        "input_ids" : torch.cat(input_ids, dim = 0),
        "attention_mask" : torch.cat(attn_mask, dim = 0),
        "labels" : torch.cat(label_ids, dim = 0),
        "input_for_generation_ids" : torch.cat(input_for_generation_ids, dim = 0),
        "conclusion" : torch.cat(conclusion_ids, dim = 0)
    }

    return model_inputs