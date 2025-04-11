#scripts/run_llama3_gguf.py
import sys
import os
import argparse
from llama_cpp import Llama

def main():
    parser = argparse.ArgumentParser(description="Run Llama 3 8B inference with GGUF")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()
    
    MODEL_PATH = "models/llama3-8b-gguf/llama-3-8b-instruct.Q4_K_M.gguf"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please run the download script first: python scripts/download_llama3_gguf.py")
        sys.exit(1)
    
    print(f"Loading model from {MODEL_PATH}...")
    print(f"Using {args.n_threads} CPU threads")
    
    # Initialize the model
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        use_mlock=True,
    )
    
    print("\nLlama 3 8B Chat (type 'exit' to quit)")
    print("-------------------------------------")
    
    chat_history = []
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible while being safe."
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Format the messages for Llama 3
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for exchange in chat_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current user query
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=512,
        )
        
        assistant_message = response["choices"][0]["message"]["content"]
        
        # Update chat history
        chat_history.append({"user": user_input, "assistant": assistant_message})
        
        print(f"\nAssistant: {assistant_message}")

if __name__ == "__main__":
    main()