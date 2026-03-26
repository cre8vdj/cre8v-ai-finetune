# 🧠 LLM Fine-Tuning — LoRA & QLoRA with PEFT

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface) ![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-orange) ![License](https://img.shields.io/badge/License-MIT-green)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cre8vdj/cre8v-ai-finetune/blob/main/finetune.ipynb)

A complete notebook for **fine-tuning large language models (LLMs)** on custom datasets using **LoRA** and **QLoRA** techniques. Achieves efficient domain adaptation with minimal GPU memory using PEFT, Transformers, and BitsAndBytes quantization.

Supports fine-tuning **Llama 2**, **Mistral**, **Falcon**, and other open-source LLMs on any instruction-following dataset.

---

## ✨ Features

- 🦙 Fine-tune **Llama-2-7B**, **Mistral-7B**, or any HuggingFace model
- 💾 **4-bit QLoRA** — fits in a free T4 GPU (Google Colab)
- 🎯 **LoRA adapters** — train only a fraction of parameters
- 📊 **WandB integration** — track training metrics in real-time
- 📄 **Custom dataset** — bring your own instruction-tuning data (JSON/JSONL)
- 💾 **Model merging** — merge LoRA weights back into base model
- 🚀 **Inference ready** — test your fine-tuned model in the same notebook
- 🚀 **Google Colab ready** — zero local GPU required

---

## 📊 Memory Requirements

| Model | Method | VRAM Required |
|-------|--------|---------------|
| Llama-2-7B | QLoRA (4-bit) | ~6 GB (Free T4) |
| Llama-2-7B | LoRA (16-bit) | ~14 GB |
| Mistral-7B | QLoRA (4-bit) | ~6 GB (Free T4) |
| Llama-2-13B | QLoRA (4-bit) | ~10 GB |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `transformers` | Model loading & training |
| `peft` | LoRA / QLoRA adapter injection |
| `bitsandbytes` | 4-bit / 8-bit quantization |
| `datasets` | Load and process training data |
| `trl` | SFTTrainer for instruction tuning |
| `wandb` | Training metrics visualization |
| `accelerate` | Multi-GPU / mixed precision |

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)

> [Open in Colab](https://colab.research.google.com/github/cre8vdj/cre8v-ai-finetune/blob/main/finetune.ipynb)

Free T4 GPU is sufficient for QLoRA fine-tuning!

### Option 2: Run Locally

```bash
# Clone
git clone https://github.com/cre8vdj/cre8v-ai-finetune.git
cd cre8v-ai-finetune

# Install dependencies
pip install transformers peft bitsandbytes datasets trl accelerate wandb

# Launch notebook
jupyter notebook finetune.ipynb
```

---

## 📁 Project Structure

```
cre8v-ai-finetune/
├── finetune.ipynb       # Main fine-tuning notebook
├── data/
│   └── sample_data.jsonl  # Example instruction dataset
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 📎 Notebook Sections

| Step | Description |
|------|-------------|
| 1. Setup | Install dependencies, login to HuggingFace |
| 2. Load Model | Load base model with 4-bit quantization |
| 3. PEFT Config | Configure LoRA / QLoRA adapter settings |
| 4. Dataset | Load and format your instruction dataset |
| 5. Train | Fine-tune using SFTTrainer |
| 6. Merge | Merge LoRA weights into base model |
| 7. Inference | Run inference with the fine-tuned model |

---

## 📄 Dataset Format

```jsonl
{"instruction": "Translate to French:", "input": "Hello, how are you?", "output": "Bonjour, comment vas-tu?"}
{"instruction": "Summarize this:", "input": "Long text here...", "output": "Short summary."}
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> Built by [cre8vdj](https://www.upwork.com/freelancers/~01fe4edd532e7be28f) · AI/ML Engineer · LLM Fine-Tuning Specialist
