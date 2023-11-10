import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def get_logger(logger_name, output_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console_handler)
    return logger


def pre_process_data(in_path, out_path):
    error_counter = 0
    counter = 0
    length_counter = 0
    with jsonlines.open(out_path, "w") as f_out:
        with open(in_path, "r") as f_in:
            while True:
                line = f_in.readline()
                if not line:
                    break
                else:
                    try:
                        item = json.loads(line)
                        if item['context_label'] == "submission":
                            f_out.write({"instruction": item["title"],
                                         "input": item["context"],
                                         "output": item["comment"]})
                            length_counter += len(item["title"].split()) + len(item["context"].split()) + len(
                                item["comment"].split())
                        else:
                            f_out.write({"instruction": item["context"],
                                         "input": "",
                                         "output": item["comment"]})
                            length_counter += len(item["context"].split()) + len(item["comment"].split())
                    except:
                        print(error_counter, item)
                        error_counter += 1
                    counter += 1
                    # if counter >= 10000:
                    #     break
    print(f"sum : {counter} ; error sum : {error_counter}")
    print(f"Avg. Length :{length_counter/(counter-error_counter)}")


def clean_data(in_path, out_path):
    error_counter = 0
    counter = 0
    length_counter = []
    import numpy as np
    with jsonlines.open(out_path, "w") as f_out:
        with open(in_path, "r") as f_in:
            while True:
                line = f_in.readline()
                if not line:
                    break
                else:
                    try:
                        item = json.loads(line)
                        instruction_str = item["instruction"]
                        input_str = item["input"]
                        output_str = item["output"]
                        if instruction_str in ["[removed]", "[deleted]", "[deleted by user]"]:
                            pass
                        if input_str in ["[removed]", "[deleted]", "[deleted by user]"]:
                            pass
                        if output_str in ["[removed]", "[deleted]", "[deleted by user]"]:
                            pass
                        item["instruction"] = clean(instruction_str, no_urls=False, lower=False, lang="en")
                        item["input"] = clean(input_str, no_urls=False, lower=False, lang="en")
                        item["output"] = clean(output_str, no_urls=False, lower=False, lang="en")

                        f_out.write(item)
                        length_counter.append(len(item["instruction"].split()) +
                                              len(item["input"].split()) +
                                              len(item["output"].split()))
                    except:
                        print(error_counter, item)
                        error_counter += 1
                    counter += 1
                    if counter >= 10000:
                        break
    length_counter = np.array(length_counter)
    print(f"sum : {counter} ; error sum : {error_counter}")
    print(f"Avg. Length :{np.mean(length_counter)}, Std. Length : {np.std(length_counter)}")
    print(f"length <= 64 : {np.sum(length_counter <= 64) / len(length_counter)}")
    print(f"length <= 128: {np.sum(length_counter <= 128) / len(length_counter)}")
    print(f"length <= 256: {np.sum(length_counter <= 256) / len(length_counter)}")
    print(f"length <= 384: {np.sum(length_counter <= 384) / len(length_counter)}")
    print(f"length <= 512: {np.sum(length_counter <= 512) / len(length_counter)}")
    print(f"length <=1024: {np.sum(length_counter <= 1024) / len(length_counter)}")


def train(
        args,
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        group_by_length: bool = True,  # faster, but produces an odd training loss curve,
):
    resume_from_checkpoint = args.resume_from_checkpoint # either training checkpoint or final adapter
    model_config = json.load(open(args.model_config_file))
    model_type = model_config['model_type']
    model_name_or_path = args.model_name_or_path
    data_path = args.data_path
    output_dir = args.output_dir
    cutoff_len = model_config['cutoff_len']
    logger = get_logger("train", output_dir)
    logger.info("args.__dict__ : {}".format(args.__dict__))
    for key, value in model_config.items():
        logger.info("{} : {}".format(key, value))
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # new_data_path = os.path.join(os.path.split(data_path)[0], "LMdata")
    # os.makedirs(new_data_path, exist_ok=True)
    # new_data_path = os.path.join(new_data_path, "text.jsonl")
    # # pre_process_data(data_path, new_data_path)
    # data_path = new_data_path

    gradient_accumulation_steps = model_config['batch_size'] // model_config[
        'per_device_train_batch_size'] if "gradient_accumulation_steps" not in model_config else model_config[
        'gradient_accumulation_steps']

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    load_in_8bit = True if args.use_lora else False
    model = None
    if model_type == "GPT2":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=os.path.join(args.cache_dir, "hub"))
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            cache_dir=os.path.join(args.cache_dir, "hub")
        )
    elif model_type.lower() == "bloom":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            cache_dir=os.path.join(args.cache_dir, "hub")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=os.path.join(args.cache_dir, "hub"))
    elif model_type.lower() == "llama":
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            cache_dir=os.path.join(args.cache_dir, "hub")
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, cache_dir=os.path.join(args.cache_dir, "hub"))
    elif model_type.lower() == "t5":
        model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            cache_dir=os.path.join(args.cache_dir, "hub")
        )
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, cache_dir=os.path.join(args.cache_dir, "hub"))
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (

                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= cutoff_len:
            result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][cutoff_len - 1] = 1

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        input_sentence = data_point["input"]
        input_text = f"### Text:\n{input_sentence.strip()}\n\n### Question:\nIs the above text steganographic or non-steganographic?\n\n### Answer:\n"
        input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
        target_text = data_point["output"] + tokenizer.eos_token
        full_prompt = input_text + target_text
        if model_type == "t5":

            tokenized_full_prompt = tokenizer(full_prompt)
            if not train_on_inputs:
                user_prompt = input_text
                tokenized_user_prompt = tokenizer(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                        user_prompt_len:
                                                                        ]
        else:
            tokenized_full_prompt = tokenize(full_prompt)
            if not train_on_inputs:
                user_prompt = input_text
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                        user_prompt_len:
                                                                        ]
        return tokenized_full_prompt


    if args.use_lora:
        model = prepare_model_for_int8_training(model)
        lora_hyperparams = json.load(open(args.lora_hyperparams_file))
        for key, value in lora_hyperparams.items():
            logger.info("{} : {}".format(key, value))
        config = LoraConfig(
            r=lora_hyperparams['lora_r'],
            lora_alpha=lora_hyperparams['lora_alpha'],
            target_modules=lora_hyperparams['lora_target_modules'] if model_config['model_type'] == "llama" else [
                "query_key_value"],
            lora_dropout=lora_hyperparams['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(config)
        model = get_peft_model(model, config)

    # block_size_10MB = 10 << 20
    data = load_dataset(path="json", data_files=data_path, num_proc=8, cache_dir=os.path.join(args.cache_dir, "datasets"))
    print(data["train"][0])
    val_set_size = int(model_config['val_set_rate'] * len(data["train"]))
    if val_set_size > 0:
        val_set_size = min(val_set_size, int(len(data['train']) * model_config['val_set_rate']))
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    logger.info("start train...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=model_config['per_device_train_batch_size'],
            per_device_eval_batch_size=model_config['per_device_train_batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=model_config['warmup_steps'],
            num_train_epochs=args.num_epoches,
            learning_rate=model_config['learning_rate'],
            fp16=True,
            logging_steps=model_config['logging_steps'],
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=model_config["eval_steps"] if val_set_size > 0 else None,
            save_steps=model_config["save_steps"],
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=args.deepspeed if not args.use_lora else None,
            group_by_length=group_by_length
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print("trainer.train")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("Save checkpointing...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.eval()
    with torch.no_grad():
        num_epochs = args.num_epoches
        with open(os.path.join(args.data_dir, "test.jsonl"), "r") as f_test, \
            jsonlines.open(os.path.join(args.data_dir, f"{os.path.basename(model_name_or_path)}-{num_epochs}-{args.specified_name}-output.jsonl"), "w") as f_out:
            while True:
                line = f_test.readline()
                if not line:
                    break
                example = json.loads(line)
                input_sentence = example["input"]
                label = example["output"]
                input_text = f"### Text:\n{input_sentence.strip()}\n\n### Question:\nIs the above text steganographic or non-steganographic?\n\n### Answer:\n"
                input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
                inputs = tokenizer([input_text], return_tensors="pt")
                predict = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=20)
                example["predict"] = tokenizer.decode(predict[0][inputs.input_ids.shape[1]:])
                f_out.write(example)



    print("\n If there's a warning about missing keys above when using lora to train, please disregard :)")
    logger.info("Training succeeded")


def convert_txt2jsonl(input_dir=""):
    stegos = open(os.path.join(input_dir, "stego.txt"),encoding='utf-8').read().split("\n")
    covers = open(os.path.join(input_dir, "cover.txt"),encoding='utf-8').read().split("\n")
    with jsonlines.open(os.path.join(input_dir, "train.jsonl"), "w") as f:
        for stego in stegos[:int(args.tr*args.ratio)]:
            if stego == "":
                continue
            else:
                f.write({"input": stego, "output": "steganographic"})
        for cover in covers[:int(args.tr)]:
            if cover == "":
                continue
            else:
                f.write({"input": cover, "output": "non-steganographic"})
    with jsonlines.open(os.path.join(input_dir, "test.jsonl"), "w") as f:
        for stego in stegos[9800:]:
            if stego == "":
                continue
            else:
                f.write({"input": stego, "output": "steganographic"})
        for cover in covers[9800:]:
            if cover == "":
                continue
            else:
                f.write({"input": cover, "output": "non-steganographic"})


# def convert_txt2jsonl(input_dir):
#     # stegos = json.loads([i for i in open(os.path.join(input_dir, "predictions_0418_embed_bit=8.json"),encoding='utf-8')])
#     # covers = json.loads([i for i in open(os.path.join(input_dir, "alpaca_data_cleaned_1000.json"),encoding='utf-8')])
#     # gpts = json.load(open(os.path.join(input_dir, "chatgpt_full.json"), encoding='utf-8'))
#     stegos = []
#     covers = []
#     gpts = []
#     with open("data_3c/alpaca_data_cleaned_1000.json", "r", encoding='utf-8') as f:
#         for jsonObj in f:
#             covers.append(json.loads(jsonObj)['output'])
#     print("loading covers:" + str(len(covers)))
#     with open("data_3c/predictions_1bit.json", "r", encoding='utf-8') as f:
#         stegos = json.load(f)
#         # tmp_stego = json.load(f)
#     # for i in range(len(stegos)):
#     #     stegos[i] = stegos[i]
#     print("loading stegos:"  + str(len(stegos)))
#     with open("data_3c/chatgpt_full.json", "r", encoding='utf-8') as f:
#         tmp_gpts=json.load(f)
#     for i in tmp_gpts:
#         gpts.append((i["output"]))
#     print("loading gpts:" + str(len(gpts)))
#     with jsonlines.open(os.path.join(input_dir, "train.jsonl"), "w") as f:
#         for stego in stegos[:800]:
#             if stego == "":
#                 continue
#             else:
#                 f.write({"input": stego, "output": "steganographic"})
#         for cover in covers[:800]:
#             if cover == "":
#                 continue
#             else:
#                 f.write({"input": cover, "output": "non-steganographic"})
#         # for gpt in gpts[:800]:
#         #     if gpt == "":
#         #         continue
#         #     else:
#         #         f.write({"input": gpt, "output": "language model"})
#     with jsonlines.open(os.path.join(input_dir, "test.jsonl"), "w") as f:
#         for stego in stegos[800:]:
#             if stego == "":
#                 continue
#             else:
#                 f.write({"input": stego, "output": "steganographic"})
#         for cover in covers[800:]:
#             if cover == "":
#                 continue
#             else:
#                 f.write({"input": cover, "output": "non-steganographic"})
#         # for gpt in gpts[800:]:
#         #     if gpt == "":
#         #         continue
#         #     else:
#         #         f.write({"input": gpt, "output": "language model"})

if __name__ == "__main__":
    i = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str,
                        default="./configs/TrainLM_llama-7b-hf.json")
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--lora_hyperparams_file", default="./configs/lora_config.json", type=str, help="Provide it when use_lora=True")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use lora")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--cache_dir", type=str, default="/data/huggingface", help="the huggingface cache path in the used machine, usually ~/.cache/huggingface")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--data_dir", type=str, default="data/data/")
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--tr", type=int, default=8000)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--specified_name", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    # args = parser.parse_args(["--use_lora"])
    args = parser.parse_args()

    convert_txt2jsonl(args.data_dir)
    args.data_path = os.path.join(args.data_dir, "train.jsonl")
    train(args)


