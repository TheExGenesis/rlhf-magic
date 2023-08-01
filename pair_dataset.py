import torch
from torch.utils.data import Dataset, random_split

def rlhf_tokenize(tokenizer, max_length, prompt):
    return tokenizer(
        f'<|startoftext|>{prompt}<|endoftext|>',
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = rlhf_tokenize(tokenizer, max_length, chosen)
            rejected_encodings_dict = rlhf_tokenize(tokenizer, max_length, rejected)
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]

def data_collator(data):
    """
    output: 
    "input_ids": all chosen input ids, then all rejected input ids
    "attention_mask": all chosen attention masks, then all rejected attention masks
    """
    return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
            'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}

