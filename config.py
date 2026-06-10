"""
Configuration file for JBShield.
"""

# Data paths
path_harmful = "data/harmful.csv"
path_harmless = "data/harmless.csv"
path_harmful_test = "data/harmful_test.csv"
path_harmless_test = "data/harmless_test.csv"
path_harmful_calibration = "data/harmful_calibration.csv"
path_harmless_calibration = "data/harmless_calibration.csv"

# Model paths
model_paths = {
    "mistral": "/root/model/Mistral-7B-Instruct-v0.2",
    "qwen2.5-7b": "/root/model/Qwen2.5-7B-Instruct",
    "deepseek-7b": "/root/model/DeepSeek-LLM-7B-Chat",
    "vicuna-7b": "/root/model/vicuna-7b-v1.5",
    "llama-3": "/root/model/Meta-Llama-3-8B",
    "mistral-sorry-bench": "/root/model/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
}
