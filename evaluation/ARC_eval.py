import sys
import os
sys.path.insert(1, os.getcwd())
from models.test_main_model_v1 import LLM_v1
from models.instruct_model import Instruct_Model_v1
from tokenizer.variants import BPE_tokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
import json

# Load model

print("Loading model...")
with open('./training/sample.json') as f:
    model = LLM_v1(json.load(f))
model = torch.compile(model, mode='default')
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
model.load_state_dict(torch.load('./models/instruct_model_v3'))
tokenizer = BPE_tokenizer()
model = model.cuda().eval()

# Get answer token IDs
# answer_tokens = []
# for letter in [" A", " B", " C", " D"]:
#     tokens = tokenizer.encode(letter)
#     answer_tokens.append(tokens[-1] if tokens else tokenizer.encode(letter.strip())[0])
# print(answer_tokens)

# print(f"Answer tokens: {answer_tokens}")

# Evaluation
total_correct = 0
total_questions = 0
results_by_subject = {}

print("\nEvaluating ARC (25-shot)...")

dataset_challenge = load_dataset("allenai/ai2_arc", name="ARC-Challenge")
dataset_easy = load_dataset("allenai/ai2_arc", name="ARC-Easy")

# Get few-shot examples
fewshot = []
if 'train' in dataset_easy:
    fewshot = list(dataset_easy['train'])[:25]
elif 'validation' in dataset_easy:
    fewshot = list(dataset_easy['validation'])[:25]
correct = 0
total = 0

for example in dataset_easy['test']:
    # Build prompt with few-shot examples
    prompt = ""
    
    for ex in fewshot:
        prompt += f"<|prompt|>Question: {ex['question']}\n"
        for i, choice in enumerate(ex['choices']['text']):
            prompt += f"{ex['choices']['label'][i]}. {choice}\n"
        prompt += f"\n<|response|>Answer: {ex['answerKey']}\n\n"
    
    # Add test question
    prompt += f"<|prompt|>Question: {example['question']}\n"
    for i, choice in enumerate(example['choices']['text']):
        prompt += f"{example['choices']['label'][i]}. {choice}\n"
    prompt += "\n<|response|>Answer: "
    
    answer_tokens = []
    if example['answerKey'] in example['choices']['label']:
        for number in example['choices']['label']:
            answer_tokens+=tokenizer.encode(number)
    else:
        for alpha in example['choices']['label']:
            answer_tokens+=tokenizer.encode(alpha)

    # Get prediction
    inputs = torch.tensor(tokenizer.encode(prompt)[-2048:], device='cuda')
    # print(inputs.shape)
    inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(inputs)
        last_token_logits = logits[0, -1]
    
    # Compare A/B/C/D logits
    ans_logits = last_token_logits[answer_tokens]
    predicted = ans_logits.argmax().item()
    
    if predicted == example['choices']['label'].index(example['answerKey']):
        correct += 1
    total += 1

accuracy = correct / total
total_correct += correct
total_questions += total
        
    # except Exception as e:
    #     tqdm.write(f"Error on {subject}: {e}")

# Final results
overall_accuracy = total_correct / total_questions

print("\n" + "="*70)
print(f"ARC-Challenge ACCURACY: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
print("="*70)

# Save results
with open('ARC-Challenge-25shot.json', 'w') as f:
    json.dump({
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'num_fewshot': 0,
    }, f, indent=2)

print("\nResults saved to arc_baseline_results.json")

# ===== GET ANSWER TOKENS =====
# def get_answer_tokens(tokenizer, choices):
#     """Get token IDs for A, B, C, D"""
#     # answer_tokens = []
    
#     # Try with space (most common after "Answer:")
#     # for letter in choices['label']:
#     #     tokens = tokenizer.encode(letter)
#     #     if len(tokens) > 0:
#     #         answer_tokens.append(tokens[-1])
    
#     # if len(answer_tokens) == 4:
#     #     print(f"Using answer tokens with space: {answer_tokens}")
#     #     return answer_tokens
    
#     # Try without space
#     answer_tokens = []
#     for letter in choices['label']:
#         tokens = tokenizer.encode(letter)
#         if len(tokens) > 0:
#             answer_tokens.append(tokens[0])
    
#     # print(f"Using answer tokens without space: {answer_tokens}")
#     return answer_tokens

# # ===== BUILD PROMPT =====
# def build_arc_prompt(question, choices, fewshot_examples=None):
#     """
#     Build ARC prompt with few-shot examples
#     """
#     prompt = ""
    
#     # Add few-shot examples
#     if fewshot_examples:
#         for ex in fewshot_examples:
#             prompt += f"Question: {ex['question']}\n"
#             for i, choice_text in enumerate(ex['choices']['text']):
#                 choice_label = ex['choices']['label'][i]
#                 prompt += f"{choice_label}. {choice_text}\n"
#             prompt += f"Answer: {ex['answerKey']}\n\n"
    
#     # Add test question
#     prompt += f"Question: {question}\n"
#     for i, choice_text in enumerate(choices['text']):
#         choice_label = choices['label'][i]
#         prompt += f"{choice_label}. {choice_text}\n"
#     prompt += "Answer:"
    
#     return prompt

# # ===== EVALUATION FUNCTION =====
# def evaluate_arc(subset="ARC-Challenge", num_fewshot=25):
#     """
#     Evaluate on ARC
#     subset: "ARC-Challenge" or "ARC-Easy"
#     """
#     print(f"\n{'='*60}")
#     print(f"Evaluating {subset} with {num_fewshot}-shot")
#     print('='*60)
    
#     # Load dataset
#     dataset = load_dataset("allenai/ai2_arc", subset)
    
#     print(f"Dataset splits: {dataset.keys()}")
#     print(f"Train size: {len(dataset['train']) if 'train' in dataset else 'N/A'}")
#     print(f"Validation size: {len(dataset['validation']) if 'validation' in dataset else 'N/A'}")
#     print(f"Test size: {len(dataset['test']) if 'test' in dataset else 'N/A'}")
    
#     # Get few-shot examples from train set
#     fewshot_examples = None
#     if num_fewshot > 0 and 'train' in dataset:
#         fewshot_examples = list(dataset['train'])[:num_fewshot]
#         print(f"\nUsing {len(fewshot_examples)} few-shot examples from train set")
    
#     # Evaluate on test set
#     test_data = dataset['test']
    
#     correct = 0
#     total = 0
#     results = []
    
#     for example in tqdm(test_data, desc=f"Evaluating {subset}"):
#         question = example['question']
#         choices = example['choices']
#         answer_key = example['answerKey']
#         answer_tokens = get_answer_tokens(tokenizer, choices=choices)

#         # Verify
#         # for i, token_id in enumerate(answer_tokens):
#         #     print(f"  {chr(65+i)}: token_id={token_id}, decodes to '{tokenizer.decode([token_id])}'")
#         # Build prompt
#         prompt = build_arc_prompt(question, choices, fewshot_examples)
        
#         # Tokenize and get logits
#         inputs = torch.tensor(tokenizer.encode(prompt)[-2048:], device='cuda')
        
#         with torch.no_grad():
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 logits = model(inputs.unsqueeze(0))
#             last_token_logits = logits[0, -1]
        
#         # Get logits for A, B, C, D
#         num_choices = len(choices['text'])
#         ans_logits = last_token_logits[answer_tokens[:num_choices]]
        
#         # Predict
#         predicted_idx = ans_logits.argmax().item()
#         predicted_label = choices['label'][predicted_idx]
        
#         # Check correctness
#         is_correct = (predicted_label == answer_key)
        
#         if is_correct:
#             correct += 1
#         total += 1
        
#         # Store result
#         results.append({
#             'question': question,
#             'choices': choices,
#             'predicted': predicted_label,
#             'correct': answer_key,
#             'is_correct': is_correct
#         })
    
#     accuracy = correct / total if total > 0 else 0
    
#     print(f"\n{'='*60}")
#     print(f"{subset} Results:")
#     print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
#     print('='*60)
    
#     return {
#         'subset': subset,
#         'accuracy': accuracy,
#         'correct': correct,
#         'total': total,
#         'num_fewshot': num_fewshot,
#         'results': results
#     }

# # ===== RUN EVALUATIONS =====
# all_results = {}

# # Evaluate ARC-Challenge
# arc_challenge_results = evaluate_arc("ARC-Challenge", num_fewshot=25)
# all_results['ARC-Challenge'] = arc_challenge_results

# # Evaluate ARC-Easy
# arc_easy_results = evaluate_arc("ARC-Easy", num_fewshot=25)
# all_results['ARC-Easy'] = arc_easy_results

# # ===== SUMMARY =====
# print("\n" + "="*60)
# print("FINAL ARC RESULTS")
# print("="*60)
# print(f"ARC-Challenge (25-shot): {all_results['ARC-Challenge']['accuracy']:.2%}")
# print(f"ARC-Easy (25-shot): {all_results['ARC-Easy']['accuracy']:.2%}")
# print("="*60)

# # Save detailed results
# with open('arc_evaluation_results.json', 'w') as f:
#     # Remove detailed results to save space
#     summary = {
#         'ARC-Challenge': {
#             'accuracy': all_results['ARC-Challenge']['accuracy'],
#             'correct': all_results['ARC-Challenge']['correct'],
#             'total': all_results['ARC-Challenge']['total'],
#             'num_fewshot': all_results['ARC-Challenge']['num_fewshot']
#         },
#         'ARC-Easy': {
#             'accuracy': all_results['ARC-Easy']['accuracy'],
#             'correct': all_results['ARC-Easy']['correct'],
#             'total': all_results['ARC-Easy']['total'],
#             'num_fewshot': all_results['ARC-Easy']['num_fewshot']
#         }
#     }
#     json.dump(summary, f, indent=2)

# print("\nResults saved to arc_evaluation_results.json")

# def build_prompt(question, choices, fewshot_examples=None):
#     prompt = ""
#     if fewshot_examples:
#         for ex in fewshot_examples:
#             prompt += f"<|prompt|>Question: {ex['question']}\n"
#             for i, choice_text in enumerate(ex['choices']['text']):
#                 label = ex['choices']['label'][i]
#                 prompt += f"{label}. {choice_text}\n"
#             prompt += f"\n<|response|>Answer: {ex['answerKey']}\n\n"
#     prompt += f"<|prompt|>Question: {question}\n"
#     for i, choice_text in enumerate(choices['text']):
#         label = choices['label'][i]
#         prompt += f"{label}. {choice_text}\n"
#     prompt += "\n<|response|>Answer:"
#     return prompt

# # ===== Likelihood scoring =====
# def score_candidate(model, tokenizer, prompt, candidate):
#     # Append candidate answer
#     full_text = prompt + f" {candidate}"
#     tokens = tokenizer.encode(full_text)
#     input_ids = torch.tensor(tokens[:-1], device='cuda').unsqueeze(0)
#     target_ids = torch.tensor(tokens[1:], device='cuda').unsqueeze(0)

#     with torch.no_grad():
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             logits = model(input_ids)
#         log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

#     token_log_probs = log_probs[0, torch.arange(target_ids.shape[1]), target_ids[0]]
#     return token_log_probs.sum().item()

# # ===== Evaluate ARC =====
# def evaluate_arc(subset="ARC-Easy", num_fewshot=25, truncate=2048):
#     print(f"\nEvaluating {subset} ({num_fewshot}-shot)...")
#     dataset = load_dataset("allenai/ai2_arc", name=subset)

#     # Select few-shot examples from train
#     fewshot_examples = None
#     if num_fewshot > 0 and 'train' in dataset:
#         fewshot_examples = list(dataset['train'])[:num_fewshot]

#     correct = 0
#     total = 0
#     results = []

#     for example in tqdm(dataset['test'], desc=f"Evaluating {subset}"):
#         question = example['question']
#         choices = example['choices']
#         answer_key = example['answerKey']

#         # Build prompt with few-shot examples
#         prompt = build_prompt(question, choices, fewshot_examples)

#         # Truncate prompt if too long
#         prompt_tokens = tokenizer.encode(prompt)
#         if len(prompt_tokens) > truncate:
#             prompt_tokens = prompt_tokens[-truncate:]
#             prompt = tokenizer.decode(prompt_tokens)

#         # Compute likelihood scores for all candidates
#         scores = [score_candidate(model, tokenizer, prompt, c) for c in choices['text']]
#         predicted_idx = scores.index(max(scores))
#         predicted_label = choices['label'][predicted_idx]

#         is_correct = predicted_label == answer_key
#         if is_correct:
#             correct += 1
#         total += 1

#         results.append({
#             'question': question,
#             'choices': choices,
#             'predicted': predicted_label,
#             'correct': answer_key,
#             'is_correct': is_correct
#         })

#     accuracy = correct / total if total > 0 else 0
#     print(f"{subset} Accuracy: {accuracy:.2%} ({correct}/{total})")

#     return {
#         'subset': subset,
#         'accuracy': accuracy,
#         'correct': correct,
#         'total': total,
#         'num_fewshot': num_fewshot,
#         'results': results
#     }

# # ===== Run evaluations =====
# all_results = {}
# all_results['ARC-Easy'] = evaluate_arc("ARC-Easy", num_fewshot=25)
# all_results['ARC-Challenge'] = evaluate_arc("ARC-Challenge", num_fewshot=25)

# # ===== Save summary =====
# summary = {
#     k: {
#         'accuracy': v['accuracy'],
#         'correct': v['correct'],
#         'total': v['total'],
#         'num_fewshot': v['num_fewshot']
#     } for k, v in all_results.items()
# }

# with open("arc_likelihood_results.json", "w") as f:
#     json.dump(summary, f, indent=2)

# print("\nResults saved to arc_likelihood_results.json")