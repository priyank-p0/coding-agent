
#formatting the test dataset for finetuning a code review model using ShareGPT format
#%%
import json

fixed = []
with open(r'T:/code/coding-agent/data/ref-train.jsonl', 'r') as f:
    for line in f:
        obj = json.loads(line)
        if 'ids' in obj:
            obj['ids'] = [str(x) for x in obj['ids']]  # force every element to string
        fixed.append(obj)

# save back
import jsonlines
with jsonlines.open(r'T:/code/coding-agent/data/ref-train-fixed.jsonl', mode='w') as writer:
    writer.write_all(fixed)


#%%
from datasets import load_dataset, Dataset
from unsloth import standardize_sharegpt
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, Value
raw_ds   = Dataset.from_json("T:/code/coding-agent/data/ref-test-fixed.jsonl")
def to_sharegpt_format(example):
    old_code = example['old_hunk'] or example['old']  # depending on quality
    comment  = example['comment']
    new_code = example['hunk'] or example['new']      # fixed code

    lang = example.get('lang', 'python')  # fallback if missing

    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a world-class code reviewer for {lang} projects."
            },
            {
                "role": "user",
                "content": f"Here is a code snippet:\n```{lang}\n{old_code}\n```\n"
                           f"Please identify issues and suggest fixes."
            },
            {
                "role": "assistant",
                "content": f"{comment}"
            },
            {
                "role": "user",
                "content": "Thank you. Please provide the corrected code snippet."
            },
            {
                "role": "assistant",
                "content": f"```{lang}\n{new_code}\n```"
            }
        ]
    }

raw_ds = raw_ds.map(to_sharegpt_format)
std_ds = standardize_sharegpt(raw_ds)   

# %%
import os

os.environ['HF_HOME'] = "T:/huggingface_cache"
hf_home = os.getenv('HF_HOME')
print(hf_home)
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
tok = AutoTokenizer.from_pretrained("unsloth/llama-3.2-3b-instruct")
get_chat_template(tok, chat_template="llama-3.2")

# 2. Format your dataset
def fmt(example):
    txt = tok.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": txt}

train_ds = std_ds.map(fmt, remove_columns=std_ds.column_names)

model, model_tok = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.2-3b-instruct",
    load_in_4bit=False,
    device_map="auto"
)

FastLanguageModel.for_inference(model)

model, _ = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.2-3b-instruct",
    load_in_4bit=True,
    device_map="auto"
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)

FastLanguageModel.train(
    model=model,
    dataset=train_ds,
    epochs=2,
    batch_size=8,
    lr=2e-4,
    val_set_size=0.02,
    train_text_field="text",
)

model.save_pretrained("llama3.2-code_review-lora")
