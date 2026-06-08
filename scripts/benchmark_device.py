#!/usr/bin/env python3
"""Benchmark CPU vs MPS/CUDA speedup for PPO training.

Usage:
    python scripts/benchmark_device.py --steps 1000
    python scripts/benchmark_device.py --steps 1000 --device cpu

Outputs:
    Time per step (ms)
    Throughput (steps/sec)
    Memory usage (MB)
"""
import argparse
import os
import sys
import time
from typing import Tuple

import torch


# Add root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _best_device() -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def benchmark_ppo(
    device: str,
    steps: int = 1000,
    n_envs: int = 4,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Train PPO for N steps on the given device.
    
    Returns:
        (time_per_step_ms, throughput_steps_per_sec, peak_memory_mb)
    """
    from config import Config
    from doom.campaign import make_campaign_env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 Benchmarking PPO on {device.upper()}")
        print(f"{'='*60}")
        print(f"  Steps: {steps:,}")
        print(f"  Parallel envs: {n_envs}")
    
    # Build environment
    cfg = Config()
    fns = [
        make_campaign_env(cfg, cfg.maps[0], rank, memory_dir=None)
        for rank in range(n_envs)
    ]
    venv = DummyVecEnv(fns)
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=cfg.frame_stack)
    
    if verbose:
        print(f"  Env: {cfg.maps[0]} × {n_envs} parallel")
        print(f"  Obs shape: {venv.single_observation_space}")
    
    # Create model
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        learning_rate=cfg.learning_rate,
        clip_range=cfg.clip_range,
        device=device,
        verbose=0,
    )
    
    # Warm up (1 internal step)
    if verbose:
        print("  Warming up...")
    model.learn(total_timesteps=n_envs * cfg.n_steps, progress_bar=False)
    
    # Benchmark
    if verbose:
        print(f"  Training {steps:,} steps...")
    
    torch.mps.empty_cache() if device == "mps" else (
        torch.cuda.empty_cache() if device == "cuda" else None
    )
    
    start = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False, reset_num_timesteps=False)
    elapsed = time.perf_counter() - start
    
    time_per_step_ms = (elapsed / steps) * 1000
    throughput = steps / elapsed
    
    # Memory estimate (rough)
    try:
        if device == "mps":
            # MPS doesn't expose memory directly; use tensor count as proxy
            peak_memory_mb = None
        elif device == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory_mb = None
    except Exception:
        peak_memory_mb = None
    
    venv.close()
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"  Time per step: {time_per_step_ms:.2f} ms")
        print(f"  Throughput: {throughput:.1f} steps/sec")
        if peak_memory_mb:
            print(f"  Peak VRAM: {peak_memory_mb:.1f} MB")
        print()
    
    return time_per_step_ms, throughput, peak_memory_mb or 0.0


def main():
    p = argparse.ArgumentParser(description="Benchmark PPO on CPU vs GPU devices.")
    p.add_argument("--steps", type=int, default=1000, help="Training steps per device.")
    p.add_argument("--device", type=str, default=None,
                   help="Device to benchmark (default: auto-detect all available).")
    p.add_argument("--n-envs", type=int, default=4, help="Parallel envs.")
    p.add_argument("--quiet", action="store_true", help="Suppress output.")
    args = p.parse_args()
    
    # Determine devices to benchmark
    devices_to_test = []
    if args.device:
        devices_to_test = [args.device]
    else:
        # Test all available
        devices_to_test.append("cpu")
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        if torch.backends.mps.is_available():
            devices_to_test.append("mps")
    
    results = {}
    
    print("\n" + "="*60)
    print("🔬 PPO DEVICE BENCHMARK")
    print("="*60)
    print(f"Testing devices: {', '.join(d.upper() for d in devices_to_test)}")
    print(f"Steps per device: {args.steps:,}")
    print("="*60)
    
    for device in devices_to_test:
        try:
            time_per_step, throughput, peak_mem = benchmark_ppo(
                device, args.steps, args.n_envs, verbose=not args.quiet
            )
            results[device] = (time_per_step, throughput, peak_mem)
        except Exception as e:
            print(f"❌ {device.upper()} failed: {e}")
            if not args.quiet:
                import traceback
                traceback.print_exc()
    
    # Summary & speedup
    if len(results) > 1:
        print("\n" + "="*60)
        print("📈 SUMMARY")
        print("="*60)
        
        baseline = results["cpu"]
        for device, (time_per_step, throughput, _) in results.items():
            speedup = baseline[0] / time_per_step
            label = "🏆 BASELINE" if device == "cpu" else f"{speedup:.1f}x faster"
            print(f"  {device.upper():6s}  {time_per_step:6.2f} ms/step  "
                  f"{throughput:6.1f} steps/sec  {label}")
        print("="*60)


if __name__ == "__main__":
    main()
