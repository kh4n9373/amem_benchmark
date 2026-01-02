#!/usr/bin/env python3
"""
A-mem Indexing Script
Process conversational data and index into A-mem memory system with full features.
Chunks conversations per-turn (user + assistant message pairs).
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

# Add A-mem to path
# sys.path.insert(0, '/home/vinhpq/mem_baseline/A-mem')
from amem.agentic_memory.memory_system import AgenticMemorySystem


def extract_turns(dialogs):
    """
    Extract turn pairs (user + assistant) from dialog sessions.
    Each turn contains user message + assistant response.
    
    Args:
        dialogs: List of dialog sessions with messages
        
    Returns:
        List of turns with combined content and metadata
    """
    turns = []
    
    for session in dialogs:
        session_id = session.get('session_id', 'unknown')
        timestamp = session.get('datetime', '')
        messages = session.get('messages', [])
        
        # Group messages into user-assistant pairs
        i = 0
        while i < len(messages):
            # Determine current message role
            current_msg = messages[i]
            current_role = current_msg.get('role', 'user')
            
            # Skip if explicitly assistant or system (we want to start turns with user/initiator)
            if current_role in ['assistant', 'Assistant', 'system', 'System', 'model', 'Model']:
                i += 1
                continue

            # Treat as user/initiator message
            user_msg = current_msg
            user_content = user_msg.get('content', '')
            user_role = current_role
            
            # Look for assistant/responder response
            assistant_content = ""
            assistant_role = ""
            
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                next_role = next_msg.get('role', 'assistant')
                
                # Check if it's a response (different role)
                if next_role != user_role:
                    assistant_content = next_msg.get('content', '')
                    assistant_role = next_role
                    i += 2  # Move past both messages
                else:
                    i += 1 # Same role, consume user message
            else:
                i += 1  # Last message
                
            # Create turn
            if user_content:
                turn_content = f"User: {user_content}"
                if assistant_content:
                    turn_content += f"\nAssistant: {assistant_content}"
                
                turns.append({
                    "content": turn_content,
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "user_role": user_role,
                    "assistant_role": assistant_role if assistant_content else None
                })
    
    return turns


def process_indexing(
    input_file: str,
    base_output_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o-mini",
    api_key: str = "dummy",
    base_url: str = None,
    evo_threshold: int = 100,
    disable_thinking: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """
    Index conversations into A-mem memory system with full features.
    
    Args:
        input_file: Path to LoCoMo/LongMemEval JSON dataset
        base_output_dir: Directory to save memory systems
        model_name: Embedding model for ChromaDB
        llm_backend: LLM backend (openai/ollama)
        llm_model: LLM model name
        api_key: API key for LLM
        base_url: Base URL for LLM API
        evo_threshold: Consolidate memories every N notes
        num_shards: Total number of shards for parallel processing
        shard_id: Current shard ID (0-indexed)
    """
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} conversations")
    print(f"Processing shard {shard_id + 1}/{num_shards}")
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Store metadata about indexed conversations
    index_metadata = []
    
    # Process each conversation
    for conv_idx, conversation in enumerate(tqdm(dataset, desc=f"Indexing (shard {shard_id}/{num_shards})")):
        # Sharding check
        if conv_idx % num_shards != shard_id:
            continue
            
        conv_id = conversation.get("conv_id", f"conv_{conv_idx}")
        dialogs = conversation.get('dialogs', [])
        
        # Extract turn pairs
        turns = extract_turns(dialogs)
        
        if not turns:
            print(f"\n⚠️  Conversation {conv_id}: No turns extracted, skipping")
            continue
        
        # Create directory for this conversation
        conv_dir = os.path.join(base_output_dir, conv_id)
        os.makedirs(conv_dir, exist_ok=True)
        
        # RESUME CAPABILITY: Check if already completed
        completed_flag = os.path.join(conv_dir, "completed.flag")
        if os.path.exists(completed_flag):
            print(f"  ⏭️  Skipping {conv_id} (already completed)")
            continue
            
        # Turn-level Resume: Load existing progress
        processed_turns_file = os.path.join(conv_dir, "processed_turns.json")
        processed_turns = set()
        if os.path.exists(processed_turns_file):
            try:
                with open(processed_turns_file, 'r') as f:
                    processed_turns = set(json.load(f))
                print(f"  Resuming {conv_id}: {len(processed_turns)} turns already processed")
            except Exception as e:
                print(f"  ⚠️  Error loading processed_turns.json: {e}")
                
        # Initialize A-mem memory system with full features
        # Check if persistent DB exists to load from it (AgenticMemorySystem doesn't load inherently, 
        # but we need to ensure we don't overwrite if we are not clearing)
        # However, AgenticMemorySystem inits with empty dicts. 
        # Since we are appending to persistent ChromaDB (on disk), we can continue seamlessly.
        # But we need to make sure we don't lose the "in-memory" state if we need it for evolution?
        # Evolution relies on `self.memories` which is in-memory.
        # IF WE CRASH AND RESTART, `self.memories` is EMPTY. 
        # Evolution will assume no history. This is a limitation unless we reload `self.memories` from ChromaDB.
        # For now, we accept this limitation for resume capability (evolution might be suboptimal for resumed turns but resumes are rare).
        
        try:
            memory_system = AgenticMemorySystem(
                model_name=model_name,
                llm_backend=llm_backend,
                llm_model=llm_model,
                evo_threshold=evo_threshold,
                api_key=api_key,
                base_url=base_url,
                disable_thinking=disable_thinking
            )
            
            # NOTE: Ideally we should reload `self.memories` from persistent ChromaDB to keep context for evolution.
            # But AgenticMemorySystem API doesn't support easy reloading yet.
            #Proceeding with "append-only" approach.

            # Add each turn to memory system
            added_count = 0
            for turn_idx, turn in enumerate(tqdm(turns, desc=f"Adding turns [{conv_id}]", leave=False)):
                timestamp = turn.get("timestamp", "")
                
                # Check if already processed (using timestamp + content hash or just unique ID if available)
                # Using timestamp + turn_idx as unique key for simplicity/robustness
                turn_key = f"{turn_idx}_{timestamp}"
                
                if turn_key in processed_turns:
                    # Skip without logging to keep output clean, update progress bar
                    continue

                try:
                    # Fail-Fast: Check for critical errors
                    # Note: AgenticMemorySystem might catch errors internally, so we need to be careful.
                    # Ideally, AgenticMemorySystem should propagate critical errors.
                    
                    # Convert timestamp to YYYYMMDDHHmm format if needed
                    timestamp = turn["timestamp"]
                    
                    # Add note with full A-mem features
                    # LLM will automatically extract keywords, context, tags
                    # Auto-linking and evolution happen automatically
                    note_id = memory_system.add_note(
                        content=turn["content"],
                        time=timestamp,
                        category=conv_id,
                        tags=[f"conv_{conv_id}", f"session_{turn['session_id']}"]
                    )
                    added_count += 1
                    
                    # --- Persistence Block (Save as you go) ---
                    # 1. Update processed tracking
                    processed_turns.add(turn_key)
                    with open(processed_turns_file, 'w') as f:
                        json.dump(list(processed_turns), f)
                        
                    # 2. Persist ChromaDB immediately
                    # We are doing this every turn to ensure "save as you go"
                    import chromadb
                    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                    
                    # Create persistent client (doing this repeatedly is checking existence mostly)
                    persistent_client = chromadb.PersistentClient(path=conv_dir)
                    embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
                    
                    persistent_collection = persistent_client.get_or_create_collection(
                        name="memories",
                        embedding_function=embedding_function
                    )
                    
                    # Sync only the new note to persistent DB
                    # We know note_id was just added.
                    # Retrieve it from in-memory collection
                    in_memory_collection = memory_system.retriever.collection
                    new_mem = in_memory_collection.get(ids=[note_id], include=["metadatas", "documents", "embeddings"])
                    
                    if new_mem and new_mem['ids']:
                        persistent_collection.add(
                            ids=new_mem["ids"],
                            documents=new_mem["documents"],
                            metadatas=new_mem["metadatas"],
                            embeddings=new_mem["embeddings"]
                        )
                    # ------------------------------------------

                except Exception as e:
                    # FAIL-FAST: Stop worker immediately on critical errors
                    error_msg = str(e).lower()
                    if "500" in error_msg or "connection" in error_msg or "internal server" in error_msg:
                        print(f"  ❌ CRITICAL ERROR (Fail-Fast): {e}")
                        sys.exit(1) # Exit immediately with error code
                    
                    print(f"  Error adding turn {turn_idx} for {conv_id}: {e}")
                    continue
            
            # (Loop finished) - Final persistence check not strictly needed if we save every turn, 
            # but we definitely need to save the final metadata.
            
            # Save memory metadata
            # Cannot pickle AgenticMemorySystem due to ChromaDB/SentenceTransformer objects
            # Instead, save configuration and let retrieval script recreate it
            metadata = {
                "conv_id": conv_id,
                "num_turns": len(turns),
                "num_indexed": added_count,
                "persist_directory": conv_dir,
                "model_name": model_name,
                "llm_model": llm_model,
                "llm_backend": llm_backend,
                "evo_threshold": evo_threshold,
                "api_key": api_key,
                "base_url": base_url,
            }
            
            # Save metadata to JSON
            metadata_file = os.path.join(conv_dir, "memory_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            index_metadata.append(metadata)
            
            # Mark as completed
            completed_flag = os.path.join(conv_dir, "completed.flag")
            with open(completed_flag, 'w') as f:
                f.write(datetime.now().isoformat())
            
            print(f"  Successfully indexed {added_count}/{len(turns)} turns")
            print(f"  ChromaDB persisted to: {conv_dir}")
            print(f"  Metadata saved to: {metadata_file}")
            
        except Exception as e:
            print(f"  ❌ Error processing conversation {conv_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save index metadata
    metadata_file = os.path.join(base_output_dir, f"index_metadata_{shard_id}.json")
    with open(metadata_file, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    
    print(f"\n✅ Indexing completed!")
    print(f"   Processed {len(index_metadata)} conversations")
    print(f"   Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Index conversations into A-mem memory system")
    parser.add_argument("input_file", type=str, help="Input JSON file with conversations")
    parser.add_argument("output_dir", type=str, help="Output directory for memory systems")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", 
                        help="Embedding model for ChromaDB")
    parser.add_argument("--llm_backend", type=str, default="openai", 
                        help="LLM backend (openai/ollama)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", 
                        help="LLM model name")
    parser.add_argument("--api_key", type=str, default="dummy", 
                        help="API key for LLM")
    parser.add_argument("--base_url", type=str, default=None, 
                        help="Base URL for LLM API")
    parser.add_argument("--evo_threshold", type=int, default=100, 
                        help="Consolidate memories every N notes")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable thinking for Qwen models")
    parser.add_argument("--num_shards", type=int, default=1, 
                        help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, 
                        help="Current shard ID (0-indexed)")
    
    args = parser.parse_args()
    
    process_indexing(
        input_file=args.input_file,
        base_output_dir=args.output_dir,
        model_name=args.model_name,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        evo_threshold=args.evo_threshold,
        disable_thinking=args.disable_thinking,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )


if __name__ == "__main__":
    main()

