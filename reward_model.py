from transformers import PretrainedConfig, AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None

# TODO: try reimplementing as a GPTNEOSequenceClassifier, benefits of hf integrations
class GPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__(PretrainedConfig())
        model = AutoModelForCausalLM.from_pretrained(config)
        self.config = model.config
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """
        input: (batch, seq, hidden)
        output: (batch)
        """
        # transformer_outputs is BaseModelOutputWithPast, 
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        batch_size = torch.arange(input_ids.size(0))
        # Anthropic learn the value on the first EOS token after the context
        eos_ind = torch.argmin(attention_mask, axis=1)


        # BaseModelOutputWithPast returns odict_keys(['last_hidden_state', 'past_key_values'])
        hidden_states = transformer_outputs['last_hidden_state']
        hidden_states = hidden_states[batch_size, eos_ind] # take the first EOS token

        return self.v_head(hidden_states).squeeze(-1)