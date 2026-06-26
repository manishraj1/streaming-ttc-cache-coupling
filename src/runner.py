import asyncio
import time
import aiohttp
import pandas as pd
import numpy as np
from src.generator import generate_poisson_stream

async def send_request(session, url, req_id, arrival_time, K):
    """Simulates a single user request with sample size K sent to vLLM."""
    # Delay execution until the simulated Poisson arrival timestamp hits
    delay = max(0, arrival_time - (time.time() - start_time))
    await asyncio.sleep(delay)
    
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "Write a step-by-step math proof for GSM8K problem: The total weight of 3 identical boxes is 15kg. What is the weight of 5 boxes?",
        "n": int(K), # Parallel reasoning chains (our primary control variable)
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False
    }
    
    actual_send_time = time.time() - start_time
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as response:
            res_json = await response.json()
            end_time = time.time() - start_time
            latency = end_time - actual_send_time
            
            return {
                "request_id": req_id,
                "simulated_arrival": arrival_time,
                "actual_arrival": actual_send_time,
                "latency": latency,
                "status": response.status
            }
    except Exception as e:
        return {
            "request_id": req_id,
            "simulated_arrival": arrival_time,
            "actual_arrival": actual_send_time,
            "latency": -1,
            "status": f"Failed: {str(e)}"
        }

async def run_streaming_benchmark(arrival_rate_lambda, K, num_prompts=30):
    """Drives a continuous Poisson stream of requests into vLLM server."""
    df_stream = generate_poisson_stream(arrival_rate_lambda, num_prompts)
    url = "http://localhost:8000"
    
    global start_time
    start_time = time.time()
    
    print(f"\n[Runner] Starting Stream Storm: λ={arrival_rate_lambda}, K={K}, Prompts={num_prompts}")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, row["request_id"], row["arrival_time"], K)
            for _, row in df_stream.iterrows()
        ]
        results = await asyncio.gather(*tasks)
        
    results_df = pd.DataFrame(results)
    print("\n--- Raw Experiment Stream Metrics ---")
    print(results_df.tail(10).to_string(index=False))
    return results_df

if __name__ == "__main__":
    # Local dry-run verification (will look for vllm port or log failure gracefully)
    try:
        asyncio.run(run_streaming_benchmark(arrival_rate_lambda=2.0, K=2, num_prompts=5))
    except KeyboardInterrupt:
        print("\nBenchmark halted.")