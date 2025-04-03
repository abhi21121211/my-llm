# My LLM - Run Local Language Models on CPU

A collection of scripts to download and run various open-source language models locally on CPU.

## Features

- Run multiple models on consumer hardware with minimal setup
- Focus on CPU execution (no GPU required)
- Optimized for memory efficiency with 4-bit quantization
- Interactive chat and code generation capabilities

## Included Models

| Model                     | Size   | Purpose          | Script                     |
| ------------------------- | ------ | ---------------- | -------------------------- |
| Tiny-GPT2                 | ~50MB  | Text generation  | `scripts/run_inference.py` |
| CodeLlama 7B 4-bit        | ~3.8GB | Code generation  | `scripts/run_codellama.py` |
| Mistral 7B Instruct 4-bit | ~4.1GB | Chat/General Q&A | `scripts/run_mistral.py`   |

## System Requirements

- Python 3.10+ (3.12 recommended)
- 16GB RAM recommended (8GB minimum)
- 10GB free disk space
- Windows/Linux/macOS compatible

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/my-llm.git
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

## Usage

### Basic Text Generation (Tiny-GPT2)

```bash
python scripts/run_inference.py
```

### Code Generation (CodeLlama)

```bash
# Download the model first (one-time)
python scripts/download_codellama.py

# Run the code generation
python scripts/run_codellama.py
```

### Chat (Mistral)

```bash
# Download the model first (one-time)
python scripts/download_mistral.py

# Start a chat session
python scripts/run_mistral.py
```

## Advanced Options

### Intel GPU Acceleration (Experimental)

For systems with Intel GPU, you can try GPU acceleration:

```bash
pip install -r requirements_intel.txt
python scripts/run_codellama_with_intel.py
```

## Project Structure

```
my-llm/
├── models/                # Downloaded model files
│   ├── tiny-gpt2/         # Tiny GPT-2 model files
│   ├── codellama-7b-python-4bit/  # CodeLlama model files
│   └── mistral-7b-instruct-4bit/  # Mistral model files
├── scripts/               # Python scripts
│   ├── download_*.py      # Model download scripts
│   └── run_*.py           # Model execution scripts
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Troubleshooting

- **Out of Memory**: Reduce the number of threads or try a smaller model
- **Slow Generation**: Increase the number of threads (up to your CPU core count)
- **Model Not Found**: Make sure to run the corresponding download script first

## License

This project is open source and available under the MIT License.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for model hosting
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for CPU inference
- [Transformers](https://github.com/huggingface/transformers) library

## Note

Large language models may sometimes generate incorrect information. Use the outputs responsibly and verify important information from reliable sources.
