import sys
import os
import torch
import argparse
import json
sys.path.insert(1, os.getcwd())
from tokenizer.variants import BPE_tokenizer, SpacyTokenizer
from models.test_main_model_v1 import LLM_v1
from models.instruct_model import Instruct_Model_v1
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
from data_processing.chunk_loader import ChunkLoader,IFT_ChunkDataset
from datasets import load_dataset, interleave_datasets, IterableDataset, Features, Value, load_from_disk
import numpy as np


def extract_instructions(example):
    example['text'] = ''
    for conv_type in example['messages']:
        if conv_type['role'] == 'user':
            example['text']+=('<|prompt|>'+conv_type['content']+'\n\n')
        else:
            example['text']+=('<|response|>'+conv_type['content']+'\n\n')
    example['text']+='<|endoftext|>'
    return example

def calculate_batch_loss(batch_input, batch_target, model, device):
    batch_input, batch_target = batch_input.to(device), batch_target.to(device)

    logits = model(batch_input)

    loss = nn.functional.cross_entropy(logits.flatten(0, 1), batch_target.flatten())

    return loss, logits

def create_chunk_dataloader(token_ids, batch_size=4, max_length=2048, stride=2048, num_workers=0) -> DataLoader:
    dataset = ChunkLoader(token_ids, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    return dataloader

def add_repetition_penalty(logits, idx_cond, device):
    # print(logits.shape)
    logits[0, idx_cond[-10:]]/=torch.arange(1.1, 1.2, 10, device=device)
    return logits

def generate_text_simple(model, idx, max_new_tokens, context_size, device, temperature_scale=0.0, top_k=None, eos_id=None):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(idx_cond) ### batch, n_tokens, vocab_size        
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # Apply softmax to get probabilities
        logits = logits[:, -1, :]
        logits = add_repetition_penalty(logits=logits, idx_cond=idx_cond, device=device)
          # (batch, vocab_size)
        if temperature_scale>0.0:
            logits = logits//temperature_scale
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # Get the idx of the vocab entry with the highest probability value
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        if idx_next == eos_id:
            break
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, mem_optimized_training, start_context, temperature=0):
    def tokenize_and_combine(batch):
        enc_text_total = []
        for sample in batch:
            # print(sample)
            enc_text = tokenizer.encode(sample['page']+'<|endoftext|>')
            enc_text_total = enc_text_total + enc_text
        return enc_text_total
    
    def tokenize_samples(data):
        for sample in data:
            sample['text'] = tokenizer.encode(sample['text'])
        return data
    val_dataset = load_dataset("EleutherAI/wikitext_document_level", name='wikitext-103-raw-v1', split='test', streaming=True).take(8192)
    val_daloader = StatefulDataLoader(val_dataset, batch_size=8192, collate_fn=tokenize_and_combine)
    # sft_dataset = load_dataset('HuggingFaceTB/smol-smoltalk', split='test', streaming=True).take(4096)
    # sft_dataset = sft_dataset.map(extract_instructions, remove_columns=['source'])
    # val_daloader = StatefulDataLoader(sft_dataset, batch_size=4096, collate_fn=tokenize_samples)
    model.eval()
    context_size = 2048 #model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    eos_id = tokenizer.encode('<|endoftext|>')
    print(eos_id)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=2048, context_size=context_size,
            temperature_scale=temperature,
            device=device,
            eos_id=eos_id[0]
        )
        val_loss = 0.0
        total_tokens = 0.0
        with torch.no_grad():
            for batch in val_daloader:
                dataloader = create_chunk_dataloader(batch)
                ind = 0
                for batch_input, batch_target in dataloader:
                    total_tokens+=batch_input.numel()
                    if mem_optimized_training:
                    # print("mem_opt_training")
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            loss, _ = calculate_batch_loss(batch_input, batch_target, model, device)
                    else:
                        loss, _ = calculate_batch_loss(batch_input, batch_target, model, device)
                    val_loss+=loss.item()
                    ind+=1
                print('Validation loss: ', val_loss/len(dataloader), total_tokens)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def main(parser:argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--config-json',
        type = str,
        required=True
    )
    parser.add_argument(
        '--tokenizer-type',
        type = str,
        choices= ['gpt2', 'cl100k_base', 'sentence_piece', 'spacy'],
        required=True
    )
    parser.add_argument(
        '--mem_opt',
        type = bool,
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

    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # mlflow.set_experiment('MLflow_quickstart')

    args = parser.parse_args()
    config = None
    assert os.path.exists(args.config_json), \
    'config path must be valid. (It should point to a valid .json file)'
    with open(args.config_json) as file:
        config = json.load(file)
    
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "cpu"
        )
    
    tokenizer = None
    if args.tokenizer_type == 'gpt2' or args.tokenizer_type == 'cl100k_base':
        tokenizer = BPE_tokenizer(args.tokenizer_type)
    elif args.tokenizer_type == 'sentencepiece':
        TODO # type: ignore
    else:
        tokenizer = SpacyTokenizer()
    
    mem_optimized_training = (args.mem_opt)
    
    model = LLM_v1(config)

    model = torch.compile(model, mode="default")
    model = Instruct_Model_v1(model, {
        'Wqkv': '',
        'out_proj': '',
        'router': '',
        'w1': '',
        'w2': '',
        'w3': ''
    }, {
        'rank': 32,
        'alpha': 32,
        'dropout':0.0
    })
    if os.path.exists('./models/instruct_model_v3'):
        state_dict = torch.load('./models/instruct_model_v3')
        model.load_state_dict(state_dict)
    print(sum([p.numel() for p in model.parameters()]))
    # print([p for p in model.named_modules()])
    model = model.to(device)
    prompt = (
        "<|prompt|>What is a Bailey bridge and how is it used by military?\n\n"
    )
    generate_and_print_sample(model=model, tokenizer=tokenizer, device=device, mem_optimized_training=mem_optimized_training, start_context=prompt, temperature=args.temperature)
    


# sample run: python ./evaluation/text_gen_eval.py --config-json=./training/sample.json --tokenizer-type=cl100k_base --mem_opt=true --temperature=0.90
# Note: run from root Celiumnet folder.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    main(parser)
