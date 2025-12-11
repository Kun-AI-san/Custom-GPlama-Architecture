# 🧠 LLM_v1 — Custom Transformer-based Language Model

LLM_v1 is a research-focused, GPT-style large language model designed for efficient, scalable training on commodity hardware (e.g., RTX 5090). It incorporates modern architectural improvements like Grouped Query Attention, SwiGLU feedforward layers, and FlashAttention support — optimized for high-throughput autoregressive training on streaming datasets.

## 🚀 Features

- **Architecture**
  - 24 transformer layers, 14 heads
  - Embedding dim: 896, Context length: 2048
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
  - Achieves ~12,000 tokens/sec on RTX 5090 with 2048 context length and FlashAttention
  - Memory-efficient training via bitsandbytes + streaming

## 🛠 Installation

```bash
git clone https://github.com/yourusername/LLM_v1
cd LLM_v1
pip install -r requirements.txt
```

## 🧪 Training

```bash
python ./training/training.py \
      --config-json=./training/sample.json \
      --attention-type=gpa \
      --tokenizer-type=cl100k_base \
      --optimizer-type=AdamW8bit_opt \
      --learning-rate=1e-4 \
      --epochs=3
```

Training is streamed in batches — no full dataset download required.

## 📦 Model Configuration (Example)

```json
{
    "vocab_size": 100277,
    "context_length": 2048,
    "emb_dim": 896,
    "n_heads": 24,
    "n_layers": 14,
    "drop_rate": 0.0,
    "qkv_bias": false,
    "n_groups": 7,
    "use_flash_attention": true
}
```

## 📈 Benchmarking

| Feature              | Value                |
|----------------------|----------------------|
| Context Length       | 2048 tokens          |
| Tokens/sec (RTX 5090)| ~12,000              |
| Precision            | bf16          |
| Optimizer            | AdamW (pytorch) |
| Dataset(s)              | fineweb-edu,stackv2 (python), openmath (streaming) |

## 📚 Acknowledgements

This model was inspired by architectural components from:
- GPT (OpenAI)
- LLama (Meta)
- GPT-NeoX
- FlashAttention v2 (HazyResearch)
- fineweb-edu dataset (HuggingFace)

## 🧩 Next Steps

- [ ] Add weight initialization from pretraining checkpoints
- [x] Integrate LoRA adapters
- [x] Add support for rotary embeddings
- [x] Inference & evaluation scripts
