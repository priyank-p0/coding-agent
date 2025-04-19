
#formatting the test dataset for finetuning a code review model
#%%
import json
from datasets import load_dataset, Dataset
from unsloth import standardize_sharegpt
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, Value
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

#%%

fixed = []
with open(r'/workspace/code/data/Code_Refinement/ref-test.jsonl', 'r') as f:
    for line in f:
        obj = json.loads(line)
        if 'ids' in obj:
            obj['ids'] = [str(x) for x in obj['ids']]  # force every element to string
        fixed.append(obj)

# save back
import jsonlines
with jsonlines.open(r'/workspace/code/data/Code_Refinement/ref-train-fixed.jsonl', mode='w') as writer:
    writer.write_all(fixed)


#%%

raw_ds = Dataset.from_json('/workspace/code/data/Code_Refinement/ref-train-fixed.jsonl')
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

#%%
import transformers, sys,peft,trl,bitsandbytes
print(peft.__version__)
# %%



tok = AutoTokenizer.from_pretrained(
    "unsloth/Llama-3.1-8B-Instruct", 
    trust_remote_code=True,
)
get_chat_template(tok, chat_template="llama-3.1")

def fmt(example):
    txt = tok.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": txt}


train_ds = std_ds.map(fmt, remove_columns=std_ds.column_names)

max_seq_length = 2048

model, _ = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",  # already quantised
    load_in_4bit = False,
    max_seq_length = max_seq_length,
)

FastLanguageModel.for_inference(model)   
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    use_gradient_checkpointing = "unsloth",
    max_seq_length = max_seq_length,
)

config = SFTConfig(
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    per_device_train_batch_size = 4,    
    gradient_accumulation_steps   = 4,  
    learning_rate = 2e-4,
    warmup_steps = 20,
    num_train_epochs = 2,
    logging_steps = 25,
    optim = "adamw_8bit",
    seed  = 42,
    output_dir = "outputs-8b",
)


trainer = SFTTrainer(
    model  = model,
    train_dataset = train_ds,
    tokenizer = tok,
    args = config,
)

trainer.train()
model.save_pretrained("llama3.1-code_review-8B-lora")
