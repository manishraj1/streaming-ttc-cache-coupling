import numpy as np
import pandas as pd

def generate_poisson_stream(arrival_rate_lambda, num_prompts, seed=42):
    """
    Generates a dataframe of synthetic streaming prompt events.
    
    Args:
        arrival_rate_lambda (float): Expected number of requests per second (λ).
        num_prompts (int): Total number of requests to simulate.
        seed (int): Fixed random seed for locked traffic parity across baselines.
    """
    np.random.seed(seed)
    
    # Inter-arrival times from an exponential distribution (1 / lambda)
    inter_arrival_times = np.random.exponential(1.0 / arrival_rate_lambda, num_prompts)
    
    # Cumulative sum to get absolute arrival timestamps starting at 0
    arrival_timestamps = np.cumsum(inter_arrival_times)
    
    df = pd.DataFrame({
        "request_id": [f"req_{i:04d}" for i in range(num_prompts)],
        "arrival_time": arrival_timestamps,
        "inter_arrival": inter_arrival_times
    })
    
    return df

if __name__ == "__main__":
    # Quick sanity test: 5 requests/sec for 10 prompts
    test_df = generate_poisson_stream(arrival_rate_lambda=5.0, num_prompts=10)
    print("--- Poisson Stream Generation Sanity Check ---")
    print(test_df.to_string(index=False))