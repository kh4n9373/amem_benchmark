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
            # Look for user message
            if i < len(messages) and messages[i].get('role') in ['user', 'User', 'Caroline', 'Melanie']:
                user_msg = messages[i]
                user_content = user_msg.get('content', '')
                user_role = user_msg.get('role', 'user')
                
                # Look for assistant response
                assistant_content = ""
                assistant_role = ""
                if i + 1 < len(messages):
                    assistant_msg = messages[i + 1]
                    assistant_role = assistant_msg.get('role', 'assistant')
                    # Check if it's an assistant/response message
                    if assistant_role not in ['user', 'User', 'Caroline', 'Melanie'] or (
                        i + 1 < len(messages) - 1 and assistant_role != user_role
                    ):
                        assistant_content = assistant_msg.get('content', '')
                        i += 2  # Move past both messages
                    else:
                        i += 1  # Only user message
                else:
                    i += 1  # Last message, no assistant response
                
                # Create turn
                if user_content:  # Only create turn if user message exists
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
            else:
                i += 1
    
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
        
        print(f"\nProcessing conversation {conv_idx + 1}/{len(dataset)}: {conv_id}")
        print(f"  Turns to add: {len(turns)}")
        
        try:
            # Initialize A-mem memory system with full features
            memory_system = AgenticMemorySystem(
                model_name=model_name,
                llm_backend=llm_backend,
                llm_model=llm_model,
                evo_threshold=evo_threshold,
                api_key=api_key,
                base_url=base_url,
                disable_thinking=disable_thinking
            )
            
            # Add each turn to memory system
            added_count = 0
            for turn_idx, turn in enumerate(tqdm(turns, desc=f"Adding turns [{conv_id}]", leave=False)):
                try:
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
                    
                except Exception as e:
                    print(f"  Error adding turn {turn_idx} for {conv_id}: {e}")
                    continue
            
            # Manually persist ChromaDB to disk
            # AgenticMemorySystem uses in-memory ChromaDB, need to save to persistent location
            print(f"\n  💾 Saving ChromaDB to disk...")
            try:
                import chromadb
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                
                # Create persistent client
                persistent_client = chromadb.PersistentClient(path=conv_dir)
                embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
                
                # Create persistent collection
                persistent_collection = persistent_client.get_or_create_collection(
                    name="memories",
                    embedding_function=embedding_function
                )
                
                # Copy all data from in-memory to persistent
                in_memory_collection = memory_system.retriever.collection
                count = in_memory_collection.count()
                
                if count > 0:
                    # Get all data from in-memory collection
                    all_data = in_memory_collection.get(
                        include=["metadatas", "documents", "embeddings"]
                    )
                    
                    # Add to persistent collection in batches
                    batch_size = 100
                    for i in range(0, len(all_data["ids"]), batch_size):
                        end_idx = min(i + batch_size, len(all_data["ids"]))
                        persistent_collection.add(
                            ids=all_data["ids"][i:end_idx],
                            documents=all_data["documents"][i:end_idx],
                            metadatas=all_data["metadatas"][i:end_idx],
                            embeddings=all_data["embeddings"][i:end_idx]
                        )
                    
                    print(f"  ✅ Saved {count} memories to persistent ChromaDB")
                else:
                    print(f"  ⚠️  No memories to save")
                    
            except Exception as e:
                print(f"  ❌ Error saving ChromaDB: {e}")
                raise
            
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

