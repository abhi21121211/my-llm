import os
import sys
import argparse
from llama_cpp import Llama

def main():
    parser = argparse.ArgumentParser(description="Run Llama 2 7B Chat with GGUF for CPU")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size for prompt processing")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()
    
    MODEL_PATH = "models/llama-gguf/llama-2-7b-chat.Q4_K_M.gguf"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please run the download script first: python scripts/download_llama_gguf.py")
        sys.exit(1)
    
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Using {args.n_threads} CPU threads")
    
    # Initialize the model
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            use_mlock=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure llama-cpp-python is installed correctly")
        print("2. Try reducing n_threads to 2 or 1")
        print("3. Free up more system memory by closing other applications")
        sys.exit(1)
    
    print("\nLlama 2 Chat (CPU-optimized) started")
    print("-------------------------------------")
    
    chat_history = []
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible while being safe."
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Format the messages for Llama 2
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for exchange in chat_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current user query
        messages.append({"role": "user", "content": user_input})
        
        print("\nThinking...", end="", flush=True)
        
        # Generate response with progressive output
        response = ""
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=512,
            stream=True,
        )
        
        # Clear the "Thinking..." message and start output
        print("\r" + " " * 10 + "\r", end="")
        print("Assistant: ", end="", flush=True)
        
        for chunk in stream:
            if chunk['choices'][0]['delta'].get('content'):
                content = chunk['choices'][0]['delta']['content']
                print(content, end="", flush=True)
                response += content
        
        print()  # Add a newline after response
        
        # Update chat history
        chat_history.append({"user": user_input, "assistant": response})

if __name__ == "__main__":
    main()