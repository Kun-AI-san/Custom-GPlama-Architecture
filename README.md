# 🧠 LLM_v1 — Custom Transformer-based Language Model

LLM_v1 is a research-focused, GPT-style large language model designed for efficient, scalable training on commodity hardware (e.g., RTX 5090). It incorporates modern architectural improvements like Grouped Query Attention, SwiGLU feedforward layers, and FlashAttention support — optimized for high-throughput autoregressive training on streaming datasets.

## 🚀 Features

- **Architecture**
  - 16 transformer layers, 16 heads
  - Embedding dim: 2048, Context length: 2048
  - Grouped Query Attention (GQA)
  - SwiGLU Feedforward with RMSNorm
  - Tied output projection head
  - No dropout for deterministic behavior

- **Custom Modules**
  - Modular attention implementation (`models.multihead_attention`)
  - Custom LayerNorm, GELU, and RMSNorm blocks
  - Optional FlashAttention v2 support (high throughput)

- **Training Setup**
  - Tokenizer: Custom BPE or SpaCy-based tokenizer
  - Dataset: `fineweb-edu` (streaming from HuggingFace)
  - Optimizers: `AdamW`, `AdamW8bit`, `GaLoreAdamW8bit`
  - Mixed Precision: AMP (bf16 preferred)
  - Gradient Checkpointing: Supported

- **Performance**
  - Achieves ~13,000 tokens/sec on RTX 5090 with 2048 context length and FlashAttention
  - Memory-efficient training via bitsandbytes + streaming

## 🛠 Installation

```bash
git clone https://github.com/yourusername/LLM_v1
cd LLM_v1
pip install -r requirements.txt
```

## 🧪 Training

```bash
python train.py \
    --model_config configs/llm_v1.json \
    --tokenizer your_tokenizer_path_or_type \
    --dataset fineweb-edu \
    --use_flash_attention \
    --mixed_precision bf16 \
    --optimizer galore_adamw_8bit \
    --streaming True
```

Training is streamed in batches — no full dataset download required.

## 📦 Model Configuration (Example)

```json
{
  "vocab_size": 100277,
  "context_length": 2048,
  "emb_dim": 2048,
  "n_heads": 16,
  "n_layers": 16,
  "drop_rate": 0.0,
  "qkv_bias": false,
  "norm_type": "rmsnorm",
  "ff_type": "swiglu",
  "tie_embeddings": true
}
```

## 🧠 Architecture Diagram

(Add a figure here using Mermaid or a hand-drawn attention block diagram.)

## 📈 Benchmarking

| Feature              | Value                |
|----------------------|----------------------|
| Context Length       | 2048 tokens          |
| Tokens/sec (RTX 5090)| ~12,000              |
| Precision            | bf16 (AMP)           |
| Optimizer            | AdamW8bit (bitsandbytes/GaLore) |
| Dataset              | fineweb-edu (streaming) |

## 📚 Acknowledgements

This model was inspired by architectural components from:
- GPT (OpenAI)
- LLama (Meta)
- GPT-NeoX
- FlashAttention v2 (HazyResearch)
- fineweb-edu dataset (HuggingFace)

## 🧩 Next Steps

- [ ] Add weight initialization from pretraining checkpoints
- [ ] Integrate LoRA adapters
- [ ] Add support for rotary embeddings
- [ ] Inference & evaluation scripts
