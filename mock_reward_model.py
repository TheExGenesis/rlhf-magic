from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import ModelOutput
from torch import nn
import torch.nn.functional as F
import torch as t
from dataclasses import dataclass
from typing import Optional, Tuple

class GPTDivergenceRewardModel(PreTrainedModel):
    """Trains with PairwiseDivergenceTrainer, since it outputs 1 valule per token"""
    def __init__(self, config):
        super().__init__(PretrainedConfig())
        vocab_size, n_embd = 50257, 768
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.transformer = nn.Linear(n_embd, n_embd, bias=False)
        self.v_head = nn.Linear(n_embd, 1, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        batch_size = input_ids.size(0)
        batch_inds = t.arange(batch_size)
        # Anthropic learn the value on the first EOS token after the context
        eos_ind = t.argmin(attention_mask, axis=1)
        embeddings = self.embedding(input_ids)
        # transformer_outputs is BaseModelOutputWithPast, 
        transformer_outputs = F.relu(self.transformer(embeddings))

        hidden_states = transformer_outputs
        return self.v_head(hidden_states).squeeze(-1)

class GPTRewardModel(PreTrainedModel):
    """Trains with PairwiseDivergenceTrainer, since it outputs 1 valule per token"""
    def __init__(self, config):
        super().__init__(PretrainedConfig())
        vocab_size, n_embd = 50257, 768
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.transformer = nn.Linear(n_embd, n_embd, bias=False)
        self.v_head = nn.Linear(n_embd, 1, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        batch_size = input_ids.size(0)
        batch_inds = t.arange(batch_size)
        # Anthropic learn the value on the first EOS token after the context
        eos_ind = t.argmin(attention_mask, axis=1)
        embeddings = self.embedding(input_ids)
        # transformer_outputs is BaseModelOutputWithPast, 
        transformer_outputs = F.relu(self.transformer(embeddings))

        hidden_states = transformer_outputs
        hidden_states = transformer_outputs[batch_inds, eos_ind] # take the first EOS token
        return self.v_head(hidden_states).squeeze(-1)