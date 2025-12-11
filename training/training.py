import torch
import torch.nn as nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets, IterableDataset, Features, Value, load_from_disk
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
import mlflow
from mlflow.models import infer_signature
from models.schedulers import get_trap_scheduler
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from smart_open import open
import matplotlib.pyplot as plt
import gc
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))

def calculate_batch_loss(batch_input, batch_target, model, device):
    batch_input, batch_target = batch_input.to(device), batch_target.to(device)

    logits = model(batch_input)

    loss = nn.functional.cross_entropy(logits.flatten(0, 1), batch_target.flatten())

    return loss

def create_chunk_dataloader(token_ids, batch_size=4, max_length=2048, stride=2048, num_workers=0) -> DataLoader:
    dataset = ChunkLoader(token_ids, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    return dataloader

def extract_text(example):
    if 'blob_id' in example:
        if example['blob_id'] is not None:
            s3_url = f"s3://softwareheritage/content/{example['blob_id']}"
            try:
                with open(s3_url, 'rb', compression='.gz', transport_params={'client': s3}) as s3bucket:
                    content = s3bucket.read()
                    content = content.decode(example['src_encoding'])
                    example['text'] = content
            except Exception as e:
                print(f'Failed to process {s3_url}: {e}')
                example['text'] = ''
    return example
    
def extract_instruction(example):
    prompt = example['prompt']
    response = example['text']
    example['text'] = "<|prompt|>"+prompt+"<|response|>"+response
    return example

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()
    torch.cuda.set_device(rank)

def clean_distributed():
    dist.destroy_process_group()

def train_multi_gpu(rank, world_size, model, optimizer, scheduler, tokenizer, mem_optimized_training, args) -> None:

    torch.manual_seed(1234)
    setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    def tokenize_and_combine(batch):
        enc_text_total = []
        for sample in batch:
            # print(sample)
            enc_text = tokenizer.encode(sample['text']+'<|endoftext|>')
            enc_text_total = enc_text_total + enc_text
        return enc_text_total

    loss_list = []
    for epoch in range(args.epochs):
        ds_fw = load_dataset("HuggingFaceTB/smollm-corpus", name='fineweb-edu-dedup', split='train', streaming=True)
        ds_cos = load_dataset("HuggingFaceTB/smollm-corpus", name='cosmopedia-v2', split='train', streaming=True)
        ds_fw = ds_fw.map(extract_text, remove_columns=['id', 'metadata'])
        ds_cos = ds_cos.map(extract_instruction, remove_columns=['prompt', 'token_length', 'audience', 'format', 'seed_data'])
        ds_python = load_dataset("Kun-AI-san/stack_v2_cpjp", name='python_dataset', split='train', streaming=True)
        ds_python = ds_python.remove_columns(['detected_liceses'])
        ds_cpp = load_dataset("Kun-AI-san/stack_v2_cpjp", name='cpp_dataset', split='train', streaming=True)
        ds_cpp = ds_cpp.remove_columns(['detected_liceses'])
        ds_java = load_dataset("Kun-AI-san/stack_v2_cpjp", name='java_dataset', split='train', streaming=True)
        ds_java = ds_java.remove_columns(['detected_liceses'])
        ds_math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        ds_math = ds_math.map(extract_text, remove_columns=['url', 'date', 'metadata'])

        dataset = interleave_datasets([ds_python, ds_fw, ds_cpp, ds_java, ds_math, ds_cos], probabilities=[0.075, 0.5, 0.075, 0.075, 0.075, 0.2], stopping_strategy='all_exhausted')

        dataset = dataset.skip(rank*9e6)

        main_dataloader = StatefulDataLoader(dataset, batch_size=9000, collate_fn=tokenize_and_combine) # type: ignore
        if os.path.exists('./data/checkpoints'):
            main_dataloader.load_state_dict(torch.load('./data/checkpoints'))
        val_dataset = list(load_dataset("HuggingFaceFW/fineweb-edu", name='sample-100BT', split='train', streaming=True).take(4096))
        val_daloader = StatefulDataLoader(val_dataset, batch_size=4096, collate_fn=tokenize_and_combine)

        if rank == 0:
            i = 0
            total_loss = 0.0
            total_acc_steps = 0
            accumulation_factor = 32
            all_tokens = 0
            train_loss_list = []
            val_loss_list = []
            tokens_seen_list = []
    
        if rank == 0 and os.path.exists('./training/val_loss_list.npy'):
            tokens_seen_list = np.load('tokens_seen_list.npy').tolist()
            train_loss_list = np.load('train_loss_list.npy').tolist()
            val_loss_list = np.load('val_loss_list.npy').tolist()
            all_tokens = tokens_seen_list[-1]
        
        for ind, batch in enumerate(main_dataloader):
            model.train()
            start_time = time.time()
            dataloader = create_chunk_dataloader(token_ids=batch)
            true_batch_loss = 0.0
            total_tokens = 0
            for batch_input, batch_target in dataloader:
                total_acc_steps+=1
                if mem_optimized_training:
                    # print("mem_opt_training")
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss = calculate_batch_loss(batch_input, batch_target, model, device)
                        loss = loss / accumulation_factor
                    loss.backward()
                else:
                    loss = calculate_batch_loss(batch_input, batch_target, model, device)
                    loss = loss / accumulation_factor
                    loss.backward()
                if total_acc_steps % accumulation_factor == 0:
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            print(f'NaNs in parameter {name}')
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f'NaNs in the gradient of {name}')
                    if torch.isnan(loss):
                        print('NaNs in loss')
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                true_batch_loss += loss.item()
                total_tokens += batch_input.numel()
                all_tokens += batch_input.numel()
            true_batch_loss/=(len(dataloader)+1e-6)
            if rank == 0:
                torch.save(main_dataloader.state_dict(), './data/checkpoints')
                torch.save(model.state_dict(), './models/main_model_v1')
                torch.save(optimizer.state_dict(), './models/optimizer_step')
                torch.save(scheduler.state_dict(), './models/scheduler_step')
            total_time = time.time() - start_time
            if rank == 0:
                print(
                    'tokens_per_sec:', (total_tokens/total_time),
                    'batch_tokens:', total_tokens,
                    'total_time:', total_time,
                    'Batch_loss:', (true_batch_loss * accumulation_factor),
                    'learning_rate:', scheduler.get_lr(),
                    'total_tokens_seen:', all_tokens,
                    'step:', ind + 1
                )
            if ind%50 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_daloader:
                        dataloader = create_chunk_dataloader(batch)
                        for batch_input, batch_target in dataloader:
                            if mem_optimized_training:
                            # print("mem_opt_training")
                                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                    loss = calculate_batch_loss(batch_input, batch_target, model, device)
                            else:
                                loss = calculate_batch_loss(batch_input, batch_target, model, device)
                            val_loss+=loss.item()
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item()/world_size

                    if rank==0:
                        print('Validation loss: ', val_loss/len(dataloader))
                        train_loss_list.append(true_batch_loss * accumulation_factor)
                        val_loss_list.append(val_loss/len(dataloader))
                        tokens_seen_list.append(all_tokens)
                        fig, ax1 = plt.subplots(figsize=(10, 6))

                        # Plot training and validation loss against epochs
                        ax1.plot(tokens_seen_list, train_loss_list, label="Training loss")
                        ax1.plot(tokens_seen_list, val_loss_list, linestyle="-.", label="Validation loss")
                        ax1.set_xlabel("Tokens seen")
                        ax1.set_ylabel("Loss")
                        ax1.legend(loc="upper right")
                        fig.tight_layout()
                        fig.savefig('progress.png')
                        plt.close('all')
                        gc.collect()
                        np.save('tokens_seen_list', tokens_seen_list)
                        np.save('train_loss_list', train_loss_list)
                        np.save('val_loss_list', val_loss_list)

            i+=1
            total_loss+=(true_batch_loss*accumulation_factor)
        if rank == 0 and os.path.exists('./data/checkpoints'):
            os.remove('./data/checkpoints')
        total_loss/=(i+1e-6)
        # mlflow.log_metric("Epoch Loss", total_loss)
        loss_list.append(total_loss)
    if rank==0:
        print('epoch loss:', loss_list)
    clean_distributed()


def train(model, device, optimizer, scheduler, tokenizer, mem_optimized_training, args) -> None:

    print("Single GPU mode")

    torch.manual_seed(1234)

    model = model.to(device=device)

    def tokenize_and_combine(batch):
        enc_text_total = []
        for sample in batch:
            # print(sample)
            enc_text = tokenizer.encode(sample['text']+'<|endoftext|>')
            enc_text_total = enc_text_total + enc_text
        return enc_text_total

    loss_list = []
    for epoch in range(args.epochs):
        ds_fw = load_dataset("HuggingFaceTB/smollm-corpus", name='fineweb-edu-dedup', split='train', streaming=True)
        ds_cos = load_dataset("HuggingFaceTB/smollm-corpus", name='cosmopedia-v2', split='train', streaming=True)
        ds_fw = ds_fw.map(extract_text, remove_columns=['id', 'metadata'])
        ds_cos = ds_cos.map(extract_instruction, remove_columns=['prompt', 'token_length', 'audience', 'format', 'seed_data'])
        ds_python = load_dataset("Kun-AI-san/stack_v2_cpjp", name='python_dataset', split='train', streaming=True)
        ds_python = ds_python.remove_columns(['detected_liceses'])
        ds_cpp = load_dataset("Kun-AI-san/stack_v2_cpjp", name='cpp_dataset', split='train', streaming=True)
        ds_cpp = ds_cpp.remove_columns(['detected_liceses'])
        ds_java = load_dataset("Kun-AI-san/stack_v2_cpjp", name='java_dataset', split='train', streaming=True)
        ds_java = ds_java.remove_columns(['detected_liceses'])
        
        ds_math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        ds_math = ds_math.map(extract_text, remove_columns=['url', 'date', 'metadata'])

        dataset = interleave_datasets([ds_python, ds_fw, ds_cpp, ds_java, ds_math, ds_cos], probabilities=[0.075, 0.5, 0.075, 0.075, 0.075, 0.2], stopping_strategy='all_exhausted')
        main_dataloader = StatefulDataLoader(dataset, batch_size=43000, collate_fn=tokenize_and_combine) # type: ignore
        if os.path.exists('./data/checkpoints'):
            main_dataloader.load_state_dict(torch.load('./data/checkpoints'))
        val_dataset = list(load_dataset("HuggingFaceFW/fineweb-edu", name='sample-100BT', split='train', streaming=True).take(4096))
        val_daloader = StatefulDataLoader(val_dataset, batch_size=4096, collate_fn=tokenize_and_combine)

        i = 0
        total_loss = 0.0
        total_acc_steps = 0
        accumulation_factor = 32
        all_tokens = 0
        train_loss_list = []
        val_loss_list = []
        tokens_seen_list = []

        if os.path.exists('./training/val_loss_list.npy'):
            tokens_seen_list = np.load('tokens_seen_list.npy').tolist()
            train_loss_list = np.load('train_loss_list.npy').tolist()
            val_loss_list = np.load('val_loss_list.npy').tolist()
            all_tokens = tokens_seen_list[-1]

        for ind, batch in enumerate(main_dataloader):
            model.train()
            start_time = time.time()
            dataloader = create_chunk_dataloader(token_ids=batch)
            true_batch_loss = 0.0
            total_tokens = 0
            for batch_input, batch_target in dataloader:
                total_acc_steps+=1
                if mem_optimized_training:
                    # print("mem_opt_training")
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        loss = calculate_batch_loss(batch_input, batch_target, model, device)
                        loss = loss / accumulation_factor
                    loss.backward()
                else:
                    loss = calculate_batch_loss(batch_input, batch_target, model, device)
                    loss = loss / accumulation_factor
                    loss.backward()
                if total_acc_steps % accumulation_factor == 0:
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            print(f'NaNs in parameter {name}')
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f'NaNs in the gradient of {name}')
                    if torch.isnan(loss):
                        print('NaNs in loss')
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                true_batch_loss += loss.item()
                total_tokens += batch_input.numel()
                all_tokens += batch_input.numel()
            true_batch_loss/=(len(dataloader)+1e-6)
            torch.save(main_dataloader.state_dict(), './data/checkpoints')
            torch.save(model.state_dict(), './models/main_model_v1')
            torch.save(optimizer.state_dict(), './models/optimizer_step')
            torch.save(scheduler.state_dict(), './models/scheduler_step')
            total_time = time.time() - start_time
            print(
                'tokens_per_sec:', (total_tokens/total_time),
                'batch_tokens:', total_tokens,
                'total_time:', total_time,
                'Batch_loss:', (true_batch_loss * accumulation_factor),
                'learning_rate:', scheduler.get_lr(),
                'total_tokens_seen:', all_tokens,
                'step:', ind + 1
            )
            if ind%50 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_daloader:
                        dataloader = create_chunk_dataloader(batch)
                        for batch_input, batch_target in dataloader:
                            if mem_optimized_training:
                            # print("mem_opt_training")
                                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                    loss = calculate_batch_loss(batch_input, batch_target, model, device)
                            else:
                                loss = calculate_batch_loss(batch_input, batch_target, model, device)
                            val_loss+=loss.item()
                        print('Validation loss: ', val_loss/len(dataloader))
                    train_loss_list.append(true_batch_loss * accumulation_factor)
                    val_loss_list.append(val_loss/len(dataloader))
                    tokens_seen_list.append(all_tokens)
                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    # Plot training and validation loss against epochs
                    ax1.plot(tokens_seen_list, train_loss_list, label="Training loss")
                    ax1.plot(tokens_seen_list, val_loss_list, linestyle="-.", label="Validation loss")
                    ax1.set_xlabel("Tokens seen")
                    ax1.set_ylabel("Loss")
                    ax1.legend(loc="upper right")
                    fig.tight_layout()
                    fig.savefig('progress.png')
                    plt.close('all')
                    gc.collect()
                    np.save('tokens_seen_list', tokens_seen_list)
                    np.save('train_loss_list', train_loss_list)
                    np.save('val_loss_list', val_loss_list)

            i+=1
            total_loss+=(true_batch_loss*accumulation_factor)
        os.remove('./data/checkpoints')
        total_loss/=(i+1e-6)
        # mlflow.log_metric("Epoch Loss", total_loss)
        loss_list.append(total_loss)
    print('epoch loss:', loss_list)

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
        '--optimizer-type',
        type = str,
        choices= ['Adam', 'AdamW', 'AdamW8bit_opt'],
        required=True
    )
    parser.add_argument(
        '--mem-opt',
        type = bool,
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
    world_size = 1
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    
    tokenizer = None
    if args.tokenizer_type == 'gpt2' or args.tokenizer_type == 'cl100k_base':
        tokenizer = BPE_tokenizer(args.tokenizer_type)
    elif args.tokenizer_type == 'sentencepiece':
        TODO # type: ignore
    else:
        tokenizer = SpacyTokenizer()
    
    model = LLM_v1(config)
    model = torch.compile(model, mode="default")

    mem_optimized_training = args.mem_opt

    if os.path.exists('./models/main_model_v1'):
        state_dict = torch.load('./models/main_model_v1')
        model.load_state_dict(state_dict)

    # with mlflow.start_run():
        # mlflow.log_params(config)
    optimizer = None
    scheduler = None

    if args.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate
        )
    elif args.optimizer_type == 'AdamW8bit_opt':
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )
    scheduler = get_trap_scheduler(
        optimizer=optimizer, total_steps=43000, num_warmup_steps=2150, base_lr=args.learning_rate,
        min_lr=1e-5
    )
    if os.path.exists('./models/optimizer_step'):
        state_dict = torch.load('./models/optimizer_step')
        optimizer.load_state_dict(state_dict)
    
    if os.path.exists('./models/scheduler_step'):
        state_dict = torch.load('./models/scheduler_step')
        scheduler.load_state_dict(state_dict)
    if world_size > 1:
        processes = []
        mp.set_start_method('spawn', force=True)
        for rank in world_size:    
            p = mp.Process(train_multi_gpu, 
                args=(rank, world_size, model, optimizer, scheduler, tokenizer, mem_optimized_training, args),
                nprocs=world_size,
                join=True
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        train(model, device, optimizer, scheduler, tokenizer, mem_optimized_training, args)

# sample run: python ./training/training.py --config-json=./training/sample.json --tokenizer-type=cl100k_base --optimizer-type=AdamW8bit_opt --mem-opt=false --learning-rate=2e-5 --epochs=1
# Note: run from root Celiumnet folder.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    main(parser)
