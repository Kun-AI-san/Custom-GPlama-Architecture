from torch.utils.data import Dataset
import torch

class ChunkLoader(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.output_ids.append(output_chunk)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.output_ids[index])
    

class IFT_ChunkDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.inputs = []
        self.targets = []
        eos_token = 100257
        for sample in token_ids:
            if len(sample['text']) < max_length:
                samp_inp = sample['text'] + [eos_token for i in range(max_length-len(sample['text']))]
                
                # sample['text'] += [eos_token for i in range(max_length-len(sample['text']))]
                samp_targ = sample['text'][1:] + [-100 for i in range((max_length+1)-len(sample['text']))]
                self.inputs.append(samp_inp)
                self.targets.append(samp_targ[:2048])
            elif len(sample['text']) > max_length:
                # for i in range(0, len(sample['text'])-max_length, stride):
                self.inputs.append(sample['text'][:max_length])
                self.targets.append(sample['text'][1:max_length]+[eos_token])
            else:
                self.inputs.append(sample['text'])
                self.targets.append(sample['text'][1:]+[-100])
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.targets[index])