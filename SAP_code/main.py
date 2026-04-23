import torch
from datasets import Dataset
import pandas as pd
from itertools import cycle
from datas import *
from model import *
from utils import *
from train import *

def test(w_lr, v_lr, v_register_layer, num_epochs):

    torch.cuda.empty_cache()
    model_path = "Llama-2-7b-chat-hf"
    
    grad_rate = 2e-5
    
    #load model
    model, lora_params, bias_params = load_lora_model(model_path, v_register_layer)
    lora_optimizer ,bias_optimizer = load_optimizer(lora_params, bias_params, w_lr, v_lr)
    tokenizer = load_tokenizer(model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    alpaca_dataset = Dataset.from_file('./dataset/alpaca/alpaca-cleaned-train.arrow')

    #split alpaca dataset for training and validation
    split_seed = 42
    validation_size = 100
    train_dataset = alpaca_dataset.shuffle(seed=split_seed).select(range(validation_size, 2000))
    validation_dataset = alpaca_dataset.shuffle(seed=split_seed).select(range(validation_size))
    train_loader = load_alpaca_data(tokenizer, train_dataset, max_token = 128, batch_size = 10)
    validation_loader = load_alpaca_data(tokenizer, validation_dataset, max_token = 128, batch_size = 10)
    
    table = pq.read_table('./dataset/justinphan/train-00000-of-00001.parquet')
    table = table.to_pandas()
    contrastive_dataset = Dataset.from_pandas(table)

    contrastive_loader = load_contrastive_data(tokenizer, contrastive_dataset, size = 48, seed = 42, max_token = 256, batch_size = 8)
    contrastive_loader = cycle(contrastive_loader)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, contrastive_loader, lora_optimizer, bias_optimizer, grad_rate, validate = True, validation_loader = validation_loader)
   
test(w_lr = 1e-4, v_lr = 0.05, v_register_layer = [3,9], num_epochs = 8)

