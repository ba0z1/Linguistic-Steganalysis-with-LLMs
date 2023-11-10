# Linguistic-Steganalysis-with-LLMs

## Prelimilaries

### 1. Virtual Environment
To run these codes, please first construct a new virtual environment via the following command:
```shell
pip install -r requirements.txt
```
Critical python packages are shown in this list:

```shell
torch         2.0.1
transformers  4.29.1
bitsandbytes  0.38.1
peft          0.3.0.dev0
```

### 2. Models
GS-llama Model can be downloaded from [https://cloud.tsinghua.edu.cn/d/55a2efbf51054cf0aedb/](https://cloud.tsinghua.edu.cn/d/55a2efbf51054cf0aedb/), this is a output directory contained the lora weights of GS-llama.

Llama Model is available in [https://huggingface.co/linhvu/decapoda-research-llama-7b-hf/tree/main](https://huggingface.co/linhvu/decapoda-research-llama-7b-hf/tree/main).

## Detect single sentence
Follow the test.ipynb, and you can use sentences in data to test the GS-llama model. 

The first step is to import the required packages.
```python
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from typing import List
import argparse, logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
import json, jsonlines
from cleantext import clean

from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
```
Then we need to load the model and pick up the lora weight.
```python
model_name_or_path = "path/to/llama"
load_in_8bit = True
device_map = "auto"
model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            # cache_dir=os.path.join(args.cache_dir, "hub")
        )
tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, 
            # cache_dir=os.path.join(args.cache_dir, "hub")
            )
```
```python
model_config = json.load(open("configs/TrainLM_llama-7b-hf.json"))
lora_hyperparams = json.load(open("configs/lora_config.json"))
target_modules = ["query_key_value"]
if model_config['model_type'] == "llama":
    target_modules = lora_hyperparams['lora_target_modules']  
config = LoraConfig(
    r=lora_hyperparams['lora_r'],
    lora_alpha=lora_hyperparams['lora_alpha'],
    target_modules=target_modules,
    lora_dropout=lora_hyperparams['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
use_lora = True
lora_weight_path = "path/to/lora/weight/adapter_model.bin"
if use_lora:
    ckpt_name = lora_weight_path
    lora_weight = torch.load(ckpt_name)
    set_peft_model_state_dict(model, lora_weight)
model.eval()
```
Finally, input the sentence and prompt to detect any stegos.
```python
input_sentence = "i guess if a film has magic i don't need it to be fluid or seamless"
input_text = f"### Text:\n{input_sentence.strip()}\n\n### Question:\nIs the above text steganographic or non-steganographic?\n\n### Answer:\n"
input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
predict = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=10, min_new_tokens=2)
output = tokenizer.decode(predict[0][inputs.input_ids.shape[1]:])
print(output)
```
You might get the following output.
```shell
Non-steganographic

###
```
In data, there are 3 sub-directories named ac, hc5, and adg. You can find stego.txt and cover.txt in them. The prompt we used is the prompt model trained.
