import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from utils import *

def compute_contrastive_gradients(model, dataloader, temperature = 1.0):
    batch = next(dataloader)
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    chosen_labels = batch["chosen_labels"].to(model.device)
    rejected_labels = batch["rejected_labels"].to(model.device)

    with autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # compute log probabilities of chosen and rejected data
        chosen_logps = -F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            chosen_labels.view(-1), 
            reduction='none'
        ).view(chosen_labels.shape).sum(dim=1)  # [batch_size]

        rejected_logps = -F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            rejected_labels.view(-1), 
            reduction='none'
        ).view(rejected_labels.shape).sum(dim=1)  # [batch_size]

        # compute contrastive loss
        log_ratio = (chosen_logps - rejected_logps) / temperature
        loss = -F.logsigmoid(log_ratio).mean()
        loss.backward()

    with torch.inference_mode():
        gradient = get_lora_grads(model)

    return gradient

def train(model, useful_loader, contrastive_loader, lora_optimizer, bias_optimizer, grad_rate, validate = False, validation_loader = None):
    torch.cuda.empty_cache()
    model.train()
    total_train_loss = 0
    
    for num_batch, batch in enumerate(useful_loader):
        print(f"Batch {num_batch + 1}/{len(useful_loader)}")
        set_optimizer(bias_optimizer, False)
        harmful_gradient = compute_contrastive_gradients(model, contrastive_loader, temperature = 1.0)#.to('cpu')#.unsqueeze(0)
        harmful_gradient = normalize_dict(harmful_gradient)
        lora_norm = get_lora_params(model, norm_only=True)
        lora_optimizer.zero_grad()

        with torch.inference_mode():
            merge_lora_parameters(model, harmful_gradient, grad_rate * lora_norm)
        
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        set_optimizer(bias_optimizer, True)
        set_optimizer(lora_optimizer, False)
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = -outputs.loss
            loss.backward()

        with torch.inference_mode():
            merge_lora_parameters(model, harmful_gradient, -grad_rate * lora_norm)

        set_optimizer(lora_optimizer, True)
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            lora_optimizer.step()
            zero_params(bias_optimizer)
            bias_optimizer.step()
            lora_optimizer.zero_grad()
            bias_optimizer.zero_grad()

        torch.cuda.empty_cache()

    if validate:
        model.eval()
        total_validation_loss = 0
        with torch.inference_mode():
            for batch in validation_loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_validation_loss += loss.item()
        torch.cuda.empty_cache()

        print(f"Validation Loss: {(total_validation_loss / len(validation_loader)):.4f}")
        
    print(f"Train Loss: {(total_train_loss / len(useful_loader)):.4f}")



