import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

def main():
    parser = argparse.ArgumentParser(description="Run Phi-3 Mini inference on CPU")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of CPU threads to use")
    args = parser.parse_args()
    
    MODEL_PATH = "models/phi3-mini-cpu"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please run the download script first: python scripts/download_phi3_cpu.py")
        sys.exit(1)
    
    print("Loading model and tokenizer...")
    
    # Set thread count for potential performance improvement
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"Using {args.num_threads} CPU threads")
    
    # Load tokenizer and model optimized for CPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Set memory optimization flags for CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        device_map="cpu",           # Force CPU
        low_cpu_mem_usage=True      # Optimize memory usage
    )
    
    # Interactive chat loop
    print("\nPhi-3 Mini Chat (type 'exit' to quit)")
    print("------------------------------------")
    print("Warning: CPU inference may be slow for initial responses.")
    
    chat_history = []
    system_prompt = "You are a helpful, respectful, and accurate assistant. Always answer as helpfully as possible."
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Format prompt with chat history
        prompt = format_prompt(system_prompt, user_input, chat_history)
        
        # Setup streaming to show progressive output
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare inputs with attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Generate in a separate thread to allow streaming
        generation_kwargs = dict(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,  # Enable sampling for temperature and top_p
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("\nPhi-3 is thinking...")
        
        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the response
        print("Phi-3: ", end="", flush=True)
        assistant_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            assistant_response += new_text
        print()  # Add newline at the end
        
        # Update chat history
        chat_history.append({"user": user_input, "assistant": assistant_response})

def format_prompt(system, user_input, chat_history):
    # Format according to Phi-3's expected format
    messages = [{"role": "system", "content": system}]
    
    # Add chat history
    for exchange in chat_history[-3:]:  # Keep last 3 exchanges for context
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})
    
    # Add current user query
    messages.append({"role": "user", "content": user_input})
    
    # Build prompt according to Phi-3 chat template
    formatted_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted_prompt += f"<|system|>\n{msg['content']}\n"
        elif msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{msg['content']}\n"
    
    # Add the final assistant prompt
    formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

if __name__ == "__main__":
    main()