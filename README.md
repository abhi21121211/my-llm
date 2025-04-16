# My LLM - Run Local Language Models on CPU

A collection of scripts to download and run various open-source language models locally on CPU.

## Features

- Run multiple powerful models on consumer hardware with minimal setup
- Focus on CPU execution (no GPU required)
- Optimized for memory efficiency with 4-bit quantization
- Interactive chat and code generation capabilities
- User-friendly model management system

## Included Models

| Model            | Size   | Purpose          | Script                        |
| ---------------- | ------ | ---------------- | ----------------------------- |
| Tiny-GPT2        | ~50MB  | Text generation  | `scripts/run_tiny_gpt2.py`    |
| Mistral 7B GGUF  | ~4.1GB | Chat/General Q&A | `scripts/run_mistral_gguf.py` |
| Phi-3 Mini (CPU) | ~4GB   | Chat/Reasoning   | `scripts/run_phi3_cpu.py`     |
| Llama 2 7B GGUF  | ~4GB   | General purpose  | `scripts/run_llama_gguf.py`   |
| Llama 3 8B GGUF  | ~4.5GB | CPU optimized    | `scripts/run_llama3_gguf.py`  |

## System Requirements

- Python 3.10+ (3.12 recommended)
- 16GB RAM recommended (8GB minimum)
- 10GB free disk space per model
- Windows/Linux/macOS compatible

## Installation

1. Clone this repository:

```bash
git clone https://github.com/abhi21121211/my-llm.git
cd my-llm
```



2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Easy Model Management

Use the model manager script for easy model installation and running:

```bash
# List all available models with details
python scripts/model_manager.py list

# Show system info and model recommendations based on your hardware
python scripts/model_manager.py sysinfo

# Download a model (e.g., phi3-mini)
python scripts/model_manager.py download phi3-cpu

# Run a model (will download if not already installed)
python scripts/model_manager.py run phi3-cpu

# Remove a model to free up disk space
python scripts/model_manager.py remove phi3-cpu
```

## Manual Usage

### Basic Text Generation (Tiny-GPT2)

```bash
python scripts/run_tiny_gpt2.py
```

### Chat with Phi-3

```bash
# Download the model first (one-time)
python scripts/download_phi3_cpu.py

# Start a chat session
python scripts/phi3_cpu.py
```

### Chat with Llama

```bash
# Download the model first (one-time)
python scripts/download_llama_gguf.py

# Start a chat session
python scripts/run_llama_gguf.py
```

> #### 🦙 Llama 3 Authentication Note:
>
> The Llama 3 8B GGUF model hosted at Hugging Face requires **user authentication** to download.
>
> Make sure you have a Hugging Face account and have accepted the model license terms here:
> [https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)
>
> Then set your Hugging Face token as an environment variable:
>
> **Linux/macOS:**
>
> ```bash
> export HUGGINGFACE_TOKEN=your_token_here
> ```
>
> **Windows (CMD):**
>
> ```cmd
> set HUGGINGFACE_TOKEN=your_token_here
> ```
>
> **Windows (PowerShell):**
>
> ```powershell
> $env:HUGGINGFACE_TOKEN="your_token_here"
> ```
>
> This token will be used to authenticate and securely download the model from Hugging Face.

## Advanced Options

Most scripts support additional parameters:

```bash
# Run with customized parameters
python scripts/run_phi3_cpu.py --temperature 0.8 --max_length 2048 --num_threads 8
```

## Memory Optimization Tips

- Run only one model at a time
- Close other memory-intensive applications
- For even more efficiency, try the GGUF model formats
- Adjust the `--num_threads` parameter to match your CPU core count
- Lower the context length to reduce memory usage

## Project Structure

```
my-llm/
├── models/                # Downloaded model files
│   ├── tiny-gpt2/         # Tiny GPT-2 model files
│   ├── llama3/            # Llama model files
│   ├── mistral/           # Mistral model files
│   ├── phi3-cpu/          # Phi-3 Mini model files
│   └── ...                # Other model files
├── scripts/               # Python scripts
│   ├── model_manager.py   # Unified model management script
│   ├── download_*.py      # Model download scripts
│   ├── run_*.py           # Model execution scripts
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Troubleshooting

- **Out of Memory**: Reduce the number of threads or try a smaller model like Phi-3 Mini
- **Slow Generation**: Increase the number of threads (up to your CPU core count)
- **Model Not Found**: Make sure to run the corresponding download script first
- **"ImportError: No module named..."**: Make sure all dependencies are installed

## License

This project is open source and available under the MIT License.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for model hosting
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for CPU inference
- [Transformers](https://github.com/huggingface/transformers) library
- The developers of all the open-source models included in this project

## Note

Large language models may sometimes generate incorrect information. Use the outputs responsibly and verify important information from reliable sources.
