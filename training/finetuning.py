import torch
import torch.nn as nn
from datasets import load_dataset, interleave_datasets
import os
import sys
sys.path.insert(1, os.getcwd())
from data_processing.chunk_loader import IFT_ChunkDataset
from torch.utils.data import DataLoader
from tokenizer.variants import BPE_tokenizer, SpacyTokenizer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from torchdata.stateful_dataloader import StatefulDataLoader
from models.test_main_model_v1 import LLM_v1
from models.schedulers import get_trap_scheduler
from models.instruct_model import Instruct_Model_v1
import bitsandbytes as bnb
import time

def extract_instructions(example):
    example['text'] = ''
    if 'messages' in example:
        for conv_type in example['messages']:
            if conv_type['role'] == 'user':
                example['text']+=('<|prompt|>'+conv_type['content']+'\n\n')
            else:
                example['text']+=('<|response|>'+conv_type['content']+'\n\n')
        example['text']+='<|endoftext|>'
    return example

def extract_QA_instructions(example):
    question = example['question']
    choices = example['choices']
    answer_key = example['answerKey']
    example['text'] = ''
    example['text'] += f"<|prompt|>Question: {question}\n"
    for i, choice_text in enumerate(choices['text']):
        label = example['choices']['label'][i]
        example['text'] += f"{label}. {choice_text}\n"
    example['text'] += f"\n<|response|>Answer: {answer_key}<|endoftext|>"
    return example

def create_batch_dataloader(batch, batch_size=4, stride=2048, max_length=2048):
    actual_batch = IFT_ChunkDataset(token_ids=batch, max_length=max_length, stride=stride)
    dataloader = DataLoader(actual_batch, batch_size=batch_size, drop_last=True)
    return dataloader

def calculate_batch_loss(model, input_ids, target_ids, device):
    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_ids.flatten())
    # print(loss)
    return loss

def supervised_fine_tuning(model, optimizer, scheduler, tokenizer, device, is_mem_optimized, epochs=1):
    def tokenize_samples(data):
        for sample in data:
            sample['text'] = tokenizer.encode(sample['text'])
        return data
    total_accumulated_steps = 0.0
    total_tokens_seen = 0.0
    for epoch in range(epochs):
        sft_core_dataset = load_dataset('HuggingFaceTB/smoltalk', name='all', split='train', streaming=True)
        sft_bench_datasets_c = load_dataset('allenai/ai2_arc', name='ARC-Challenge', split='train', streaming=True)
        sft_bench_datasets_e = load_dataset('allenai/ai2_arc', name='ARC-Easy', split='train', streaming=True)
        sft_core_dataset = sft_core_dataset.map(extract_instructions, remove_columns=['source'])
        sft_bench_datasets_c = sft_bench_datasets_c.map(extract_instructions, remove_columns=['source'])
        sft_bench_datasets_e = sft_bench_datasets_e.map(extract_QA_instructions, remove_columns=['id'])
        sft_dataset = interleave_datasets([sft_core_dataset, sft_bench_datasets_c, sft_bench_datasets_e], probabilities=[0.9967, 0.0011, 0.0022], stopping_strategy='all_exhausted', seed=42)
        sft_dataloader = StatefulDataLoader(sft_dataset, batch_size=2048, collate_fn=tokenize_samples)
        batch_loss_list = []
        total_token_list = []
        for master_batch in sft_dataloader:
            batch_loader = create_batch_dataloader(batch=master_batch, max_length=2048)
            # print(master_batch)
            batch_loss = 0.0
            accumulation_factor = 32
            start_time = time.time()
            batch_tokens = 0.0
            for i, (input_ids, target_ids) in enumerate(batch_loader):
                total_accumulated_steps+=1
                if is_mem_optimized:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss = calculate_batch_loss(model, input_ids=input_ids, target_ids=target_ids, device=device)
                        loss = loss / accumulation_factor
                else:
                    loss = calculate_batch_loss(model, input_ids=input_ids, target_ids=target_ids, device=device)
                    loss = loss / accumulation_factor
                loss.backward()
                if total_accumulated_steps%accumulation_factor==0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                batch_loss += loss.item()/len(batch_loader)*accumulation_factor
                total_tokens_seen += input_ids.numel()
                batch_tokens += input_ids.numel()
                batch_time = time.time()-start_time
            print('Batch_loss: ', batch_loss, 'batch_time: ', batch_time, 'batch_tokens: ', batch_tokens, 'tokens/sec: ', batch_tokens/batch_time, 'total_tokens: ', total_tokens_seen, scheduler.get_last_lr())
            batch_loss_list.append(batch_loss)
            total_token_list.append(total_tokens_seen)
            Wqkv = {}
            out_proj = {}
            w1 = {}
            w2 = {}
            w3 = {}
            for i, block in enumerate(model.foundation_model.trf_blocks):
                Wqkv[i] = block.multihead_attention.Wqkv.state_dict()
                out_proj[i] = block.multihead_attention.out_proj.state_dict()
                w1[i] = block.router.w1.state_dict()
                w2[i] = block.router.w2.state_dict()
                w3[i] = block.router.w3.state_dict()
            torch.save(model.state_dict(), './models/instruct_model_v3')
            torch.save(Wqkv, './models/Wqkv')
            torch.save(out_proj, './models/out_proj')
            torch.save(w1, './models/w1')
            torch.save(w2, './models/w2')
            torch.save(w3, './models/w3')
            torch.save(model.foundation_model.out_head.state_dict(), './models/out_head')

def main(args:argparse.ArgumentParser):
    parsed_args = args.parse_args()
    print(parsed_args)
    cfg = None
    with open(parsed_args.sample_json) as f:
        cfg = json.load(f)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    model = LLM_v1(cfg=cfg)
    model = torch.compile(model, mode = 'default')

    if os.path.exists(parsed_args.model_location):
        state_dict = torch.load(parsed_args.model_location)
        model.load_state_dict(state_dict=state_dict)
    else:
        raise 'Please provide correct model location.'
    
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
    model = model.to(device)

    tokenizer = None
    if parsed_args.tokenizer_type == 'gpt2' or parsed_args.tokenizer_type == 'cl100k_base':
        tokenizer = BPE_tokenizer(parsed_args.tokenizer_type)
    elif parsed_args.tokenizer_type == 'sentencepiece':
        TODO # type: ignore
    else:
        tokenizer = SpacyTokenizer()

    optimizer = None
    if parsed_args.optimizer_type=='Adam':
        optimizer = torch.optim.Adam(model.parameteres(), lr=parsed_args.learning_rate)
    elif parsed_args.optimizer_type=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=parsed_args.learning_rate)
    elif parsed_args.optimizer_type=='AdamW8bit':
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=parsed_args.learning_rate)
    else:
        optimizer = torch.optim.ASGD(model.parameters(), lr=parsed_args.learning_rate)

    opt_state = './models/fine_opt_state'

    if os.path.exists(opt_state):
        optimizer_state_dict = torch.load(opt_state)
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = get_trap_scheduler(optimizer=optimizer, total_steps=25000, num_warmup_steps=400, base_lr=parsed_args.learning_rate, min_lr=4e-6)
    sch_state = './models/fine_sch_state'

    if os.path.exists(sch_state):
        scheduler_state = torch.load(sch_state)
        scheduler.load_state_dict(scheduler_state)

    is_mem_optimized = parsed_args.mem_opt

    print(sum([p.numel() for p in model.parameters()]))

    supervised_fine_tuning(model, optimizer, scheduler, tokenizer, device, is_mem_optimized, epochs=parsed_args.epochs)

    # for p, pn in model.named_parameters():
    #     # if p == '_orig_mod.trf_blocks.22.multihead_attention.inner_attn':
    #     print(p, pn.shape)

    

    # print(model.parameters()['_orig_mod.trf_blocks.22.multihead_attention.Wqkv.weight'].shape)


    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--sample-json",
        type=str,
        required=True
    )
    args.add_argument(
        "--model-location",
        type=str,
        required=True
    )
    args.add_argument(
        "--learning-rate",
        type=float,
        required=True
    )
    args.add_argument(
        "--mem-opt",
        type=bool,
        required=True
    )
    args.add_argument(
        "--optimizer-type",
        type=str,
        choices=["Adam", "AdamW", "AdamW8bit"],
        required=True
    )
    args.add_argument(
        '--tokenizer-type',
        type=str,
        choices=['gpt2', 'cl100k_base', 'sentence_piece', 'spacy'],
        required=True
    )
    args.add_argument(
        "--epochs",
        type=int,
        required=True
    )
    main(args=args)