from torch.utils.data import DataLoader
from datasets import Dataset
import pyarrow.parquet as pq
    
class LabeledDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.input_ids = encoded_dataset["input_ids"]
        self.attention_mask = encoded_dataset["attention_mask"]
        self.labels = encoded_dataset["labels"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

class ContrastiveDataset(Dataset):
    def __init__(self, encoded_dataset):
        self.input_ids = encoded_dataset["input_ids"]
        self.attention_mask = encoded_dataset["attention_mask"]
        self.chosen_labels = encoded_dataset["chosen_labels"]
        self.rejected_labels = encoded_dataset["rejected_labels"]

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "chosen_labels": self.chosen_labels[idx],
            "rejected_labels": self.rejected_labels[idx]
        }

def load_alpaca_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 128, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    def alpaca_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ins + inp},
                {"role": "assistant", "content": out} 
            ]

            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
            input_texts.append(input_text)
            label_texts.append(full_text)
        
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )

        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        labels = full_encoded["input_ids"]
        
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len] = -100

        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
        
    encoded_dataset = dataset.map(alpaca_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader

def load_advbench_data(tokenizer, dataset: Dataset, size = None, seed = None, max_token = 128, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))

    def advbench_preprocess_function(examples):
        input_texts = []
        label_texts = []
        for input, output in zip(examples["prompt"], examples["target"]):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input},
                {"role": "assistant", "content": output}
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            input_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
            input_texts.append(input_text)
            label_texts.append(full_text)
        
        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        full_encoded = tokenizer(
            label_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        labels = full_encoded["input_ids"]
        
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in model_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                labels[i, :input_len] = -100
        
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
        
    encoded_dataset = dataset.map(advbench_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'labels'])
    dataset = LabeledDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader

def load_contrastive_data(tokenizer, dataset, size = None, seed =None, max_token = 128, batch_size = 16):
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))

    def ic_preprocess_function(examples):
        input_texts = []
        chosen_texts = []
        rejected_texts = []
        for input, chosen, rejected in zip(examples['prompt'], examples['llama3_output'], examples['response']):
            input_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input}
            ]
            chosen_messages = input_messages + [{"role": "assistant", "content": chosen}]
            rejected_messages = input_messages + [{"role": "assistant", "content": rejected}]

            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)  # 只到user为止
            full_chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            full_rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            
            input_texts.append(input_text)
            chosen_texts.append(full_chosen_text)
            rejected_texts.append(full_rejected_text)

        encoded_inputs = tokenizer(
            input_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        encoded_full_chosen = tokenizer(
            chosen_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        chosen_labels = encoded_full_chosen["input_ids"]

        encoded_full_rejected = tokenizer(
            rejected_texts,
            truncation=True,
            padding="max_length",
            max_length=max_token,
            return_tensors="pt",
            add_special_tokens=False
        )
        rejected_labels = encoded_full_rejected["input_ids"]
        
        input_lengths = [len(seq[seq != tokenizer.pad_token_id]) for seq in encoded_inputs["input_ids"]]
        for i, input_len in enumerate(input_lengths):
            if input_len < max_token:
                chosen_labels[i, :input_len] = -100
                rejected_labels[i, :input_len] = -100
        
        chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
        rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100

        return {'input_ids': encoded_inputs['input_ids'],
                'attention_mask': encoded_inputs['attention_mask'],
                'chosen_labels': chosen_labels,
                'rejected_labels': rejected_labels
                 }
    
    encoded_dataset = dataset.map(ic_preprocess_function, remove_columns=dataset.column_names, batched = True)
    encoded_dataset.set_format(type="torch", columns=["input_ids",
            "attention_mask",
            "chosen_labels",
            "rejected_labels"])
    dataset = ContrastiveDataset(encoded_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory = True)
    return loader


