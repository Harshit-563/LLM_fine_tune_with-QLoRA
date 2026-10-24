# Fine-Tuning Qwen3-4B with QLoRA for Medical Question Answering

Fine-tune **Qwen3-4B** for medical question answering using **QLoRA, 4-bit quantization, and supervised fine-tuning (SFT)**. This project demonstrates an end-to-end workflow for preparing medical question-answer pairs, configuring parameter-efficient fine-tuning, training a language model, and comparing its responses against the base model.

The implementation uses Hugging Face Transformers, PEFT, TRL, and bitsandbytes. Experiments were conducted on a Tesla T4 GPU with approximately 14.6 GB of reported memory.

> **Important:** This is an experimental learning project, not a clinically validated medical AI system. The evaluation set is very small, and the measured improvement is modest.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [Model and Dataset](#model-and-dataset)
- [How It Works](#how-it-works)
- [QLoRA Configuration](#qlora-configuration)
- [Training Configuration](#training-configuration)
- [Evaluation and Results](#evaluation-and-results)
- [Getting Started](#getting-started)
- [Running the Notebook](#running-the-notebook)
- [Project Workflow](#project-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Overview

Large language models can answer questions across many domains, but their responses depend on their pretrained knowledge, prompting, and training data.

This project explores whether parameter-efficient fine-tuning can adapt Qwen3-4B to a medical question-answering task using the MedQuAD dataset.

Instead of updating every parameter of the base model, the project uses **Quantized Low-Rank Adaptation (QLoRA)**. The model is loaded using 4-bit NormalFloat quantization, while LoRA adapters provide trainable parameters for supervised fine-tuning.

The notebook follows this workflow:

1. Check GPU availability.
2. Download the pretrained Qwen3-4B model.
3. Configure 4-bit quantization.
4. Load and inspect the MedQuAD dataset.
5. Prepare training and evaluation examples.
6. Format examples using the model's chat template.
7. Configure LoRA adapters.
8. Fine-tune the model using TRL's `SFTTrainer`.
9. Evaluate the base and fine-tuned models.
10. Inspect low-scoring responses to identify weaknesses.

## Objectives

- Understand practical QLoRA-based fine-tuning of a pretrained language model.
- Reduce fine-tuning memory requirements through 4-bit quantization.
- Prepare question-answer pairs for causal language model training.
- Apply parameter-efficient fine-tuning with PEFT.
- Use supervised fine-tuning with Hugging Face TRL.
- Compare model responses before and after fine-tuning.
- Identify evaluation limitations and areas for further improvement.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Tensor operations and model execution |
| Hugging Face Transformers | Model and tokenizer loading |
| Hugging Face Hub | Downloading the pretrained model |
| Hugging Face Datasets | Dataset loading and preparation |
| bitsandbytes | 4-bit model quantization |
| PEFT | LoRA configuration and parameter-efficient training |
| TRL | Supervised fine-tuning using `SFTTrainer` |
| CUDA | GPU-accelerated training and inference |
| Qwen3-4B | Pretrained base language model |
| MedQuAD | Medical question-answering dataset |

## Model and Dataset

### Base Model

**Model:** `Qwen/Qwen3-4B`

The notebook downloads the pretrained model from Hugging Face and loads it with a 4-bit quantization configuration.

The tokenizer's chat template is used to format prompts and training examples consistently with the model's expected conversation structure.

### Dataset

**Dataset:** `Hmehdi515/MedQuad`

The notebook loads the dataset from CSV files hosted on Hugging Face:

- Training source: `medquad_train.csv`
- Validation source: `medquad_val.csv`

The source dataset contains the following fields:

| Field | Description |
|---|---|
| `sample` | Sample identifier |
| `question` | Medical question |
| `answer` | Corresponding answer |

The notebook prepares a smaller subset for the experiment.

### Dataset Preparation

The preprocessing pipeline:

1. Loads the training and validation CSV files.
2. Removes surrounding whitespace from questions and answers.
3. Filters out empty questions.
4. Keeps examples with answers between 20 and 160 words.
5. Selects 96 training examples and 8 evaluation examples.
6. Shuffles the training subset using seed `42`.
7. Formats examples as system, user, and assistant messages.
8. Renders conversations with the Qwen tokenizer's chat template.

The resulting experimental dataset contains:

| Split | Examples |
|---|---:|
| Training | 96 |
| Evaluation | 8 |
| **Total** | **104** |

The source dataset is substantially larger, but the notebook intentionally uses a small subset for this experiment.

## How It Works

### 1. GPU Verification

The notebook checks whether CUDA is available and stops execution if no GPU is detected.

The recorded environment was:

- GPU: NVIDIA Tesla T4
- Reported GPU memory: approximately 14.6 GB

Actual memory requirements and performance can vary with the environment, library versions, and sequence lengths.

### 2. Four-Bit Quantization

The pretrained model is loaded using `BitsAndBytesConfig`.

The notebook configures:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.bfloat16`
- `bnb_4bit_use_double_quant=True`

**Why use quantization?**

Loading a multi-billion-parameter model in full precision can require substantial GPU memory. Four-bit quantization reduces the memory footprint of the quantized model weights, making fine-tuning more accessible on constrained hardware.

NF4 is a quantization format designed for normally distributed model weights. Double quantization further reduces the memory overhead associated with quantization constants.

The notebook then prepares the quantized model for k-bit training using PEFT's `prepare_model_for_kbit_training`.

### 3. Chat Formatting

Each training example is converted into a conversation containing three roles:

- **System:** Defines how medical questions should be answered.
- **User:** Contains the medical question.
- **Assistant:** Contains the reference answer.

The system instruction used in the notebook is:

> Answer the medical question clearly and concisely. When a question requires personal medical advice, recommend consulting a qualified clinician.

The conversation is rendered using `tokenizer.apply_chat_template()` with thinking disabled.

This teaches the model to produce answers in the intended conversational format rather than treating the question and answer as unrelated text.

### 4. LoRA Adapter Configuration

The project uses PEFT's `LoraConfig` to configure the trainable adapter parameters.

```python
from peft import LoraConfig

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

| Parameter | Value | Purpose |
|---|---|---|
| `r` | `8` | LoRA rank |
| `lora_alpha` | `16` | Adapter scaling parameter |
| `lora_dropout` | `0.05` | Dropout applied within LoRA |
| `bias` | `"none"` | Does not train bias parameters through this configuration |
| `target_modules` | `"all-linear"` | Targets linear modules according to PEFT's supported configuration |
| `task_type` | `"CAUSAL_LM"` | Specifies causal language modeling |

LoRA introduces trainable low-rank updates instead of requiring full fine-tuning of all base-model parameters. Combined with quantization, this reduces the resource requirements of the training process.

## Training Configuration

The notebook uses TRL's `SFTTrainer` with the following configuration:

```python
from trl import SFTConfig

training_config = SFTConfig(
    output_dir="/tmp/qwen-medquad",
    learning_rate=0.0002,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=384,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    bf16=True,
    gradient_checkpointing=True,
    dataset_text_field="text",
    seed=42,
)
```

### Configuration Explained

| Parameter | Value | Explanation |
|---|---|---|
| Learning rate | `0.0002` | Learning rate specified for training |
| Epochs | `2` | Number of passes through the training dataset |
| Per-device batch size | `2` | Number of examples processed per device step |
| Gradient accumulation | `4` | Accumulates gradients across four steps before an optimizer update |
| Maximum sequence length | `384` | Maximum sequence length used by the trainer |
| Logging frequency | Every step | Training loss is logged at each step |
| Checkpoint saving | Disabled | `save_strategy="no"` |
| Precision | BF16 | Enables bfloat16 training where supported |
| Gradient checkpointing | Enabled | Trades additional computation for reduced activation memory |
| Random seed | `42` | Supports reproducibility of applicable random operations |

The trainer is initialized with the quantized base model, the training dataset, and the LoRA configuration. It then performs supervised fine-tuning on the formatted conversation text.

### Recorded Training Result

The notebook completed two epochs and reported:

```text
Training loss: 1.7388
Training steps: 24
Epochs: 2
```

Training loss measures how well the model fits the training objective. It does not, by itself, establish the accuracy, safety, or clinical reliability of generated answers.

## Evaluation and Results

The notebook evaluates the pretrained model before fine-tuning and then evaluates the fine-tuned model using the same eight evaluation examples.

### Evaluation Metric

The notebook implements a token-overlap F1 score:

1. Converts predictions and reference answers to lowercase.
2. Extracts word-like tokens using a regular expression.
3. Counts overlapping tokens, accounting for repeated words.
4. Calculates precision and recall.
5. Computes their harmonic mean.
6. Averages the score across evaluation examples.

This metric measures lexical overlap between a generated answer and its reference. It does not directly measure medical correctness or factual reliability.

### Results

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Evaluation examples | 8 | 8 |
| Average token F1 | 25.96 | 26.10 |
| Change in token F1 | — | +0.14 points |

**Observed change:** The average token F1 increased by approximately **0.14 percentage points** after fine-tuning.

This is a small improvement, and the evaluation sample is too limited to establish that the fine-tuned model generalizes better to medical questions overall.

### Error Analysis

The notebook also inspects low-scoring responses to identify potential weaknesses.

For example, one of the weakest evaluated responses concerned treatments for cystinuria. The generated answer provided a general description of the condition and some treatment-related information, while the reference emphasized resources about diagnosis and management.

This illustrates an important limitation of generative question answering: a response can be broadly related to a question without matching the reference's specific content or emphasis.

## Getting Started

### Requirements

The notebook was executed in a GPU-enabled Python environment. To reproduce the experiment, use an environment with:

- Python and a compatible PyTorch installation
- A CUDA-capable NVIDIA GPU
- A compatible CUDA/PyTorch setup
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- bitsandbytes
- Hugging Face Hub

The notebook's recorded environment used a Tesla T4 GPU. Other GPUs may require adjustments to precision, batch size, or sequence length.

### Installation

The following is a starting point for setting up the Python dependencies. It is not a pinned environment specification from the notebook; compatible package versions may need to be selected for your system.

```bash
pip install torch
pip install -U transformers datasets accelerate peft trl
pip install -U "bitsandbytes>=0.46.1"
pip install -U huggingface_hub
```

Install a PyTorch build compatible with your CUDA environment before running the notebook. The commands above do not guarantee compatibility with every GPU or operating system.

### Download the Model and Dataset

The notebook uses the Hugging Face Hub to download `Qwen/Qwen3-4B` and loads the MedQuad CSV files from the dataset repository.

The first run requires an internet connection and enough storage for the model and dataset caches.

A Hugging Face access token may be configured if needed for account-specific access or higher download limits. Do not commit access tokens or other credentials to the repository.

## Running the Notebook

The original workflow is notebook-based.

1. Open the project notebook in Jupyter or a compatible hosted notebook environment.
2. Select a GPU runtime.
3. Install or update the required dependencies.
4. Execute the GPU availability check.
5. Download the pretrained model and dataset.
6. Run the dataset preparation and formatting steps.
7. Load the model with the QLoRA quantization configuration.
8. Evaluate the base model.
9. Configure LoRA and run supervised fine-tuning.
10. Evaluate the fine-tuned model.
11. Review the generated responses and low-scoring examples.

**Important:** The notebook uses temporary paths such as `/tmp/tensortonic` and `/tmp/qwen-medquad`. These paths may not persist after a hosted runtime is reset.

The recorded training configuration sets `save_strategy="no"`, so it does not save regular training checkpoints. If you want to reuse the fine-tuned adapter later, add an explicit adapter-saving step and verify that the saved files can be reloaded.

## Project Workflow

```text
Qwen3-4B Base Model
        |
        v
Load Tokenizer and Model
        |
        v
4-Bit NF4 Quantization
        |
        v
MedQuAD Dataset
        |
        v
Filter and Select Examples
        |
        v
Format Chat Conversations
        |
        v
Baseline Evaluation
        |
        v
Configure LoRA Adapters
        |
        v
Supervised Fine-Tuning
        |
        v
Fine-Tuned Model Evaluation
        |
        v
Compare Scores and Inspect Errors
```

## Limitations

This experiment is a useful demonstration of parameter-efficient fine-tuning, but several limitations should be considered.

- **Small training subset:** Only 96 examples were used for training, despite the larger source dataset.
- **Small evaluation set:** The reported score is based on just eight examples and may not represent broader performance.
- **Limited metric:** Token-level F1 rewards lexical overlap, not necessarily medical correctness, reasoning quality, or factual consistency.
- **No demonstrated clinical validation:** The experiment does not establish suitability for diagnosis, treatment decisions, or patient care.
- **No demonstrated broad generalization:** Results on a small subset cannot establish performance across medical specialties or patient populations.
- **Checkpoint persistence:** Regular checkpoint saving is disabled in the recorded training configuration.
- **Hardware and software dependency:** Reproduction may require changes for different GPUs, operating systems, and package versions.
- **Reproducibility:** The seed is specified, but exact results can still vary with software versions, hardware, and execution settings.

## Future Improvements

Potential directions for extending the project include:

- Increase the training subset and evaluate on a larger, more representative set.
- Use a fixed, held-out evaluation dataset that is separate from training data selection.
- Compare multiple LoRA ranks, learning rates, and training durations.
- Save and reload the adapter to verify reproducibility.
- Compare token F1 with additional metrics suited to question answering.
- Conduct structured error analysis for incomplete, irrelevant, or unsupported answers.
- Evaluate factuality and safety with appropriate expert review.
- Compare the fine-tuned model against the base model across multiple medical topics.
- Pin dependency versions and document the hardware and runtime environment.
- Explore more systematic hyperparameter tuning and validation.

These are proposed improvements, not features already demonstrated by the current notebook.

## Disclaimer

This project is intended for educational and research purposes only. The model may generate incorrect, incomplete, outdated, or misleading medical information.

It has not been established as a medical device or clinically validated decision-support system. Do not use its outputs as a substitute for diagnosis, treatment, emergency care, or advice from a qualified healthcare professional.

## Acknowledgements

This project builds on the following open-source technologies and resources:

- [Qwen](https://huggingface.co/Qwen/Qwen3-4B) — pretrained language model
- [MedQuAD dataset](https://huggingface.co/datasets/Hmehdi515/MedQuad) — medical question-answer data
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — model loading and inference
- [PEFT](https://github.com/huggingface/peft) — parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) — supervised fine-tuning tools
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — quantization support

---

**Key takeaway:** This project demonstrates an end-to-end QLoRA fine-tuning workflow for Qwen3-4B on a small medical question-answering subset. The measured token-F1 increase is modest, making larger-scale evaluation and further experimentation the natural next steps.
