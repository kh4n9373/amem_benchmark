#!/usr/bin/env python3
"""
Parallel Indexing Manager for A-mem
- Manages multiple indexing workers
- Implements Fault Tolerance: Fails fast if ANY worker dies (kill all strategy)
- Streams output from workers
"""

import argparse
import subprocess
import sys
import time
import os
import signal
from typing import List, Tuple

def run_parallel_indexing(
    args,
    script_path: str,
):
    """
    Run indexing workers in parallel and monitor them.
    If ANY worker fails, kill all others and exit.
    """
    print(f"🚀 Starting Parallel Indexing Manager")
    print(f"   Workers: {args.max_workers}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output:  {args.output_dir}")
    
    # List to store (process, worker_id, log_file_handle)
    workers: List[Tuple[subprocess.Popen, int, str]] = []
    
    try:
        # Start all workers
        for worker_id in range(args.max_workers):
            cmd = [
                "python3", script_path,
                args.dataset,
                args.output_dir,
                "--model_name", args.embedding_model,
                "--llm_model", args.llm_model,
                "--api_key", args.api_key,
                "--evo_threshold", str(args.evo_threshold),
                "--num_shards", str(args.max_workers),
                "--shard_id", str(worker_id),
            ]
            
            if args.base_url:
                cmd.extend(["--base_url", args.base_url])
            
            if args.disable_thinking:
                cmd.append("--disable_thinking")
            
            # Setup logging
            log_dir = args.log_dir if args.log_dir else "worker_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"worker_{worker_id}.log")
            
            print(f"   [Worker {worker_id}] Starting... Log: {log_file}")
            
            # We open the file for writing
            f = open(log_file, 'w')
            
            # Start process
            # Note: We don't pipe stdout/stderr to main process to avoid buffering issues and interleaving.
            # Users should check the individual log files or tail them.
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            workers.append((proc, worker_id, f))
        
        print("\n👀 Monitoring workers... (Press Ctrl+C to stop)")
        
        # Monitor loop
        failed = False
        while True:
            all_done = True
            for proc, worker_id, _ in workers:
                return_code = proc.poll()
                
                if return_code is None:
                    # Still running
                    all_done = False
                elif return_code != 0:
                    # FAILED!
                    print(f"\n❌ [Worker {worker_id}] FAILED with exit code {return_code}!")
                    failed = True
                    break
                else:
                    # Finished successfully (we could print this, but better to wait until all done)
                    pass
            
            if failed:
                break
            
            if all_done:
                print("\n✅ All workers completed successfully!")
                break
            
            time.sleep(1) # Check every second
            
    except KeyboardInterrupt:
        print("\n🛑 Received KeyboardInterrupt. Stopping all workers...")
        failed = True
        
    finally:
        # Cleanup
        if failed:
            print("⚠️  Terminating all workers due to failure/interruption...")
            for proc, worker_id, f in workers:
                if proc.poll() is None: # If still running
                    print(f"   Killing worker {worker_id}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                
                # Close file handle
                if not f.closed:
                    f.close()
            sys.exit(1)
        else:
            # Just close file handles
            for _, _, f in workers:
                if not f.closed:
                    f.close()


def main():
    parser = argparse.ArgumentParser(description="Parallel Indexing Manager")
    parser.add_argument("dataset", help="Input dataset path")
    parser.add_argument("output_dir", help="Output directory")
    
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--llm_model", default="gpt-4o")
    parser.add_argument("--api_key", default="dummy")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--evo_threshold", type=int, default=100)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--log_dir", default=None)
    
    args = parser.parse_args()
    
    # Find the worker script (assumed to be in same dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "amem_process_index.py")
    
    if not os.path.exists(worker_script):
        print(f"❌ Worker script not found: {worker_script}")
        sys.exit(1)
        
    run_parallel_indexing(args, worker_script)

if __name__ == "__main__":
    main()
