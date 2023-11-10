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
0
Llama Model is available in [https://huggingface.co/linhvu/decapoda-research-llama-7b-hf/tree/main](https://huggingface.co/linhvu/decapoda-research-llama-7b-hf/tree/main).

## Detect single sentence
Follow the test.ipynb, and you can use sentences in data to test the GS-llama model. 

In data, there are 3 sub-directories named ac, hc5, and adg. You can find stego.txt and cover.txt in them. The prompt we used is the prompt model trained.
