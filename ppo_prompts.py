# %%
import gzip
import json


def load_gzip_data(path):
    with gzip.open(path, "rb") as f:
        # Read each line from the file
        data = [json.loads(line) for line in f]
    return data
train_path = "hh-rlhf2/helpful-base/train.jsonl.gz"
test_path = "hh-rlhf2/helpful-base/test.jsonl.gz"
train_data = load_gzip_data(train_path)
test_data = load_gzip_data(test_path)

# %%

def get_assistant_prompts(dialogue):
    assistant_prompts = []
    assistant_marker = "Assistant: "
    dialogue_split = dialogue.split(assistant_marker)
    for i in range(1, len(dialogue_split)):
        prompt = assistant_marker.join(dialogue_split[:i])+assistant_marker
        assistant_prompts.append(prompt)
    return assistant_prompts
print(len(get_assistant_prompts(train_data[0]['chosen'])))

def convert_dataset_to_prompts(data):
    list_of_prompts = [get_assistant_prompts(x['chosen']) for x in data]
    return [item for sublist in list_of_prompts for item in sublist]

def load_ppo_prompts(path):
    # path = "hh-rlhf2/helpful-base/train.jsonl.gz"
    data = load_gzip_data(path)
    return convert_dataset_to_prompts(data)