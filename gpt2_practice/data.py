from cProfile import label
import torch
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class Syllogistic(Dataset):  
    def __init__(self, args, data, tokenizer, max_length, kfold_idx = None):
        self.data_path = os.path.join(args.data_path, data)
        self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
        self.raw_data = self.raw_data[self.raw_data["Syllogistic relation"] == "yes"].reset_index()
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        
        self.tokenizer = tokenizer
        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token
        self.prem_connect = tokenizer.additional_special_tokens[0] # '<|and|>'
        self.conc = tokenizer.additional_special_tokens[1] #'<|so|>'
        self.input_pad_id = tokenizer.pad_token_id
        self.label_pad_id = -100
        
        self.prem1 = self.raw_data["Premise 1"].to_list()
        self.prem2 = self.raw_data["Premise 2"].to_list()
        self.label = self.raw_data["Conclusion"].to_list()
        
        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in enumerate(self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in enumerate(self.prem2) if num in kfold_idx]
            self.label = [sent for num, sent in enumerate(self.label) if num in kfold_idx]

        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"

        
        self.input_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"+ label + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
        self.label_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"+ label + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
        self.input_for_generation_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"  for p1, p2 in zip(self.prem1, self.prem2)]
        self.conclusion_text = [label for label in self.label]
    
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

        # self.input_ids = []
        # self.attn_masks = []
        # self.label_ids =[]
        # self.labels = []
        
    
        # for row in range(len(self.raw_data)):
        #     encodings_input = tokenizer(self.bos + self.prem1[row] + self.prem_connect + self.prem2[row] + self.conc, 
        #                                truncation = True, max_length= max_length, padding="max_length")
        #     encodings_label = tokenizer(self.prem1[row] + self.prem_connect + self.prem2[row] + self.conc + self.label[row]+self.eos, 
        #                                truncation = True, max_length= max_length, padding="max_length")
        #     self.input_ids.append(torch.tensor(encodings_input['input_ids']))
        #     self.label_ids.append(torch.tensor(encodings_label['input_ids']))
        #     self.attn_masks.append(torch.tensor(encodings_input['attention_mask']))


    # def __getitem__(self, idx):
    #     return self.input_ids[idx], self.label_ids[idx], self.attn_masks[idx]