import torch
import torch.nn as nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from datasets import load_dataset
import sys
import os
sys.path.insert(1, os.getcwd())
from models.test_main_model_v1 import LLM_v1
from tokenizer.variants import BPE_tokenizer, SpacyTokenizer
from data_processing.chunk_loader import ChunkLoader
import bitsandbytes as bnb
import argparse
import json
import time
import tqdm

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature_scale=0.0, top_k=None, eos_id=None):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  
        if temperature_scale>0.0:
        # Apply softmax to get probabilities
            logits = logits//temperature_scale
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        if idx_next == eos_id:
            break
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calculate_batch_loss(batch_input, batch_target, model, device) -> nn.functional.cross_entropy:
    batch_input, batch_target = batch_input.to(device), batch_target.to(device)

    logits = model(batch_input)

    loss = nn.functional.cross_entropy(logits.flatten(0, 1), batch_target.flatten())

    return loss

def create_chunk_dataloader(token_ids, batch_size=3, max_length=2048, stride=2048, num_workers=0) -> DataLoader:
    dataset = ChunkLoader(token_ids, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    return dataloader


def train(model, device, optimizer, tokenizer, mem_optimized_training, args) -> None:
    def tokenize_and_combine(batch):
        enc_text_total = []
        for sample in batch:
            enc_text = tokenizer.encode(sample['text'])
            enc_text_total = enc_text_total + enc_text
        return enc_text_total
    
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name='sample-10BT', split='train', streaming=True)
    main_dataloader = StatefulDataLoader(dataset, batch_size=1024, collate_fn=tokenize_and_combine, num_workers=8)
    
    if os.path.exists('./data/checkpoints'):
        main_dataloader.load_state_dict(torch.load('./data/checkpoints'))
    
    scaler = torch.amp.GradScaler(device=device.type)
    loss_list = []
    for epoch in range(args.epochs):
        model.train()
        i = 0
        for batch in tqdm.tqdm(main_dataloader):
            start_time = time.time()
            dataloader = create_chunk_dataloader(token_ids=batch)
            true_batch_loss = 0.0
            total_tokens = 0
            for batch_input, batch_target in dataloader:
                if not mem_optimized_training:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        loss = calculate_batch_loss(batch_input, batch_target, model, device)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = calculate_batch_loss(batch_input, batch_target, model, device)
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters()], 1.0)
                true_batch_loss += loss.item()
                total_tokens += batch_input.numel()
            true_batch_loss/=(len(dataloader)+1e-6)
            torch.save(main_dataloader.state_dict(), './data/checkpoints')
            torch.save(model.state_dict(), './models/main_model_v1')
            total_time = time.time() - start_time
            print('tokens per sec:', (total_tokens/total_time), 'total tokens:', total_tokens, 'total time:', total_time)
        loss_list.append(true_batch_loss)
    print('epoch loss:', loss_list)

def main(parser:argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--config-json',
        type = str,
        required=True
    )
    parser.add_argument(
        '--attention-type',
        type = str,
        choices= ['mha', 'gpa', 'mla'],
        required=True
    )
    parser.add_argument(
        '--tokenizer-type',
        type = str,
        choices= ['gpt2', 'cl100k_base', 'sentence_piece', 'spacy'],
        required=True
    )
    parser.add_argument(
        '--optimizer-type',
        type = str,
        choices= ['Adam', 'AdamW', 'AdamW8bit_opt'],
        required=True
    )
    parser.add_argument(
        '--learning-rate',
        type = float,
        required=True
    )
    parser.add_argument(
        '--epochs',
        type = int,
        required=True
    )
    parser.add_argument(
        '--temperature',
        type = float,
        required=False
    )
    parser.add_argument(
        '--top_k',
        type = int,
        required=False
    )
    
    args = parser.parse_args()
    config = None
    assert os.path.exists(args.config_json), \
    'config path must be valid. (It should point to a valid .json file)'
    with open(args.config_json) as file:
        config = json.load(file)
    
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "rocm:0" if torch.rocm.is_available()
        else "cpu"
        )
    
    tokenizer = None
    if args.tokenizer_type == 'gpt2' or args.tokenizer_type == 'cl100k_base':
        tokenizer = BPE_tokenizer(args.tokenizer_type)
    elif args.tokenizer_type == 'sentencepiece':
        TODO
    else:
        tokenizer = SpacyTokenizer()
    
    mem_optimized_training = (args.optimizer_type=='AdamW8bit_opt')
    
    if mem_optimized_training:
        model = LLM_v1(config).half()
    else:
        model = LLM_v1(config)
    
    model.to(device=device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    if os.path.exists('./models/main_model_v1'):
        state_dict = torch.load('./models/main_model_v1')
        model.load_state_dict(state_dict)
    
    optimizer = None
    
    if mem_optimized_training:
        target_modules = ['attention', 'mlp']
        galore_params = []

        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(
                        target_key in module_name
                        for target_key in target_modules
                    ):
                continue
            galore_params.append(module.weight)
        
        id_galore_params = [id(param) for param in galore_params]

        non_galore_params = [
            param for param in model.parameters()
            if id(param) not in id_galore_params
        ]

        optimizer_dict = {}

        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = bnb.optim.AdamW8bit(
                        [p], lr=args.learning_rate, weight_decay=0.1
                    )
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=args.learning_rate, weight_decay=0.1
                    )
        
        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
    
    else:
        if args.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate
            )
    train(model, device, optimizer, tokenizer, mem_optimized_training, args)

# sample run: python ./training/training.py --config-json=./training/sample.json --attention-type=gpa --tokenizer-type=cl100k_base --optimizer-type=AdamW8bit_opt --learning-rate=1e-7 --epochs=1
# Note: run from root Celiumnet folder.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    main(parser)
