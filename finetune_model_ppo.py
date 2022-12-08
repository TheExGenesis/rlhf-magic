# %%
import sys
sys.path.append("trlx")
# %%
# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, GPTNeoForSequenceClassification
import os
import yaml

import trlx
import torch
from typing import List
from trlx.data.configs import TRLConfig
import gzip
import json
# from reward_model import GPTRewardModel
from mock_reward_model import GPTRewardModel
from transformers import PretrainedConfig
from ppo_prompts import load_ppo_prompts
from pair_dataset import rlhf_tokenize, data_collator
from trlx.model.nn.ppo_models import PPOConfig
from trlx.trlx import train

# %%




model_name = "EleutherAI/gpt-neo-125M"
default_config = yaml.safe_load(open("ppo_config.yml"))

def load_reward_model(path):
    # Load a pretrained model and tokenizer
    model = GPTRewardModel(model_name)
    model.load_state_dict(torch.load(path))
    return model 

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main(reward_model_path, hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1


    model_name = "EleutherAI/gpt-neo-125M"
    model = load_reward_model(reward_model_path)
    tokenizer = load_tokenizer(model_name)

    def reward_model_fn(samples):
        # TODO inefficient to be making many dicts and unmaking them again
        tokenized_samples = [rlhf_tokenize(tokenizer=tokenizer, max_length=1024, prompt=sample) for sample in samples]
        tokenized_batch = {'input_ids': torch.cat([f["input_ids"] for f in tokenized_samples]),
            'attention_mask': torch.cat([f["attention_mask"] for f in tokenized_samples])}
        return model(**tokenized_batch)


    # TODO: reimplement this when reward model is a pretrained model
    # pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task="text-classification"
    # )

    # Take few words off of movies reviews as prompts
    prompts = load_ppo_prompts("hh-rlhf2/helpful-base/train.jsonl.gz")

    model = train(
        reward_fn=reward_model_fn,
        prompts=prompts[:int(len(prompts)*0.9)],
        eval_prompts=prompts[(-int(len(prompts)*0.1)):],
        config=config,
    )

# %%
mock_model_path = "/home/genesis/Documents/Projects/rlhf-magic/ckpts/hh-rlhf2/helpful-base/mock-eos-gpt-neo/checkpoint-360/pytorch_model.bin"
big_LLM_model_path = "/home/genesis/Documents/Projects/rlhf-magic/ckpts/colab/colab-trained-reward-model/checkpoint-360/pytorch_model.bin"
if __name__ == "__main__":
    main(reward_model_path=mock_model_path)

