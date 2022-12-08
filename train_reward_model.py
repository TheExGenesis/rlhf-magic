# %%
# import sys
# sys.path.append("reward_modeling")
# %%
import gzip
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch as t
from torch.utils.data import random_split
from einops import rearrange
from fancy_einsum import einsum
# from reward_model import GPTRewardModel
from mock_reward_model import GPTRewardModel
from pair_dataset import PairwiseDataset, data_collator
from pair_trainer import PairwiseTrainer


def load_gzip_data(path):
    with gzip.open(path, "rb") as f:
        # Read each line from the file
        data = [json.loads(line) for line in f]
    return data


# %%
def main(train_path, test_path, model_name, dataset_name="dataset"):
    output_dir=f'ckpts/{dataset_name}/mock-eos-gpt-neo'
    # Open the file in a gzip-compressed manner
    train_data = load_gzip_data(train_path)
    test_data = load_gzip_data(test_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTRewardModel(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
    dataset = PairwiseDataset(train_data[:100], tokenizer, max_length = 1024)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=4, logging_steps=100, save_strategy="epoch",
                                    per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
                                    weight_decay=0.01, logging_dir="./logs", fp16=True, bf16=False, learning_rate=5e-6, save_total_limit=1)
    trainer = PairwiseTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=data_collator)
    trainer.train()

    t.save(model.state_dict(), f"{output_dir}/torch_save/model.bin")


if __name__ == "__main__":
    dataset_name = "hh-rlhf2/helpful-base"
    train_path = "hh-rlhf2/helpful-base/train.jsonl.gz"
    test_path = "hh-rlhf2/helpful-base/test.jsonl.gz"
    model_name = "EleutherAI/gpt-neo-125M"
    main(train_path, test_path, model_name, dataset_name=dataset_name)
