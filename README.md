# QSP: Query-Adaptive Subspace Projection for Inference-Time Jailbreak Defense

## Requirements

The minimum hardware requirement is two GPUs with at least 24GB VRAM each (e.g., RTX 3090 or RTX 4090). For optimal performance, we recommend a setup with 4 RTX 4090 GPUs (24GB VRAM each) or 1 A100 GPUs (80GB VRAM each). 

The code for `QSP` runs with Python 3 and requires Pytorch. We recommend using `Anaconda` or `miniconda` for python. Our code has been tested with `python=3.12.8` and `torch=2.5.1` on linux. First, create a conda environment activate it.


## Our Dataset

Our dataset is located in `./data`. The jailbreak prompts are located in `./data/jailbreak`, while the harmful and harmless prompts can be found in `./data/harmful{}.csv` and `./data/harmless{}.csv`, respectively. 


## Model Preparation

We selected five target LLMs and one judge LLM, as detailed in the table below:

| Model name                                     | Link                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Mistral-7B-Instruct-v0.2                       | [:hugs:[Huggingface link]](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| Llama-2-7b-chat-hf                             | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Meta-Llama-3-8B-Instruct                       | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| vicuna-7b-v1.5                                 | [:hugs:[Huggingface link]](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| ft-mistral-7b-instruct-v0.2-sorry-bench-202406 | [:hugs:[Huggingface link]](https://huggingface.co/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406) |


## QSP-D

Run the following commands to evaluate the jailbreak detection performance of `QSP-D` on the five LLMs:

```shell
chmod +x ./evaluate_projection_detection.sh
./evaluate_projection_detection.sh
```

The results are saved in `/logs/QSP-D_{model_name}.log`. We have also provided the logs from our runs in the same directory.


## QSP-M

Run the following commands to evaluate the jailbreak mitigation performance of `QSP-M` on the five LLMs:

```shell
chmod +x ./evaluate_projection_mitigation.sh
./evaluate_projection_mitigation.sh
```

The results are saved in `/logs/QSP-M.log`. We have also provided the logs from our runs in the same directory.

