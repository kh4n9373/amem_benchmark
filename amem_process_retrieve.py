#!/usr/bin/env python3
"""
A-mem Retrieval Script
Retrieve relevant memories for questions using A-mem's search_agentic() method.
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from tqdm.auto import tqdm

# Add A-mem to path
# sys.path.insert(0, '/home/vinhpq/mem_baseline/A-mem')
from amem.agentic_memory.memory_system import AgenticMemorySystem


def process_retrieval(
    input_file: str,
    memory_dir: str,
    output_file: str,
    top_k: int = 100,
):
    """
    Retrieve memories for questions using A-mem's search_agentic() method.
    
    Args:
        input_file: Path to LoCoMo/LongMemEval JSON dataset with questions
        memory_dir: Directory containing indexed memory systems
        output_file: Output file for retrieval results
        top_k: Number of memories to retrieve per question
    """
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} conversations")
    print(f"Retrieving top-{top_k} memories per question")
    
    results = []
    total_questions = 0
    processed_questions = 0
    
    # Process each conversation
    for conv_idx, conversation in enumerate(tqdm(dataset, desc="Retrieving")):
        conv_id = conversation.get("conv_id", f"conv_{conv_idx}")
        questions = conversation.get("qas", [])
        
        if not questions:
            continue
        
        total_questions += len(questions)
        
        # Load memory metadata for this conversation
        conv_dir = os.path.join(memory_dir, conv_id)
        metadata_file = os.path.join(conv_dir, "memory_metadata.json")
        
        if not os.path.exists(metadata_file):
            print(f"\n⚠️  Memory metadata not found for {conv_id}, skipping")
            continue
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load persisted ChromaDB
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            
            # Create persistent client
            persistent_client = chromadb.PersistentClient(path=conv_dir)
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=metadata.get("model_name", "all-MiniLM-L6-v2")
            )
            collection = persistent_client.get_collection(
                name="memories",
                embedding_function=embedding_function
            )
            
            # Process each question
            for q_idx, question in enumerate(questions):
                question_id = question.get("question_id", f"{conv_id}_q{q_idx}")
                question_text = question.get("question", "")
                answer = question.get("answer", "")
                evidences = question.get("evidences", [])
                category = question.get("category")
                
                if not question_text:
                    continue
                
                try:
                    # Retrieve using ChromaDB query
                    search_results = collection.query(
                        query_texts=[question_text],
                        n_results=top_k,
                        include=["metadatas", "documents", "distances"]
                    )
                    
                    # Format retrieved chunks
                    retrieved_chunks = []
                    if search_results and 'ids' in search_results and search_results['ids'] and len(search_results['ids'][0]) > 0:
                        for i, doc_id in enumerate(search_results['ids'][0]):
                            meta = search_results['metadatas'][0][i] if i < len(search_results['metadatas'][0]) else {}
                            document = search_results['documents'][0][i] if i < len(search_results['documents'][0]) else ""
                            distance = search_results['distances'][0][i] if i < len(search_results['distances'][0]) else 0.0
                            
                            # Parse JSON strings back to lists/dicts
                            keywords = meta.get('keywords', '[]')
                            if isinstance(keywords, str):
                                try:
                                    keywords = json.loads(keywords) if keywords else []
                                except:
                                    keywords = []
                            
                            tags = meta.get('tags', '[]')
                            if isinstance(tags, str):
                                try:
                                    tags = json.loads(tags) if tags else []
                                except:
                                    tags = []
                            
                            retrieved_chunks.append({
                                "id": doc_id,
                                "content": meta.get('content', document),
                                "score": 1.0 - distance,  # Convert distance to similarity score
                                "keywords": keywords,
                                "tags": tags,
                                "context": meta.get('context', ''),
                                "timestamp": meta.get('timestamp', ''),
                                "category": meta.get('category', ''),
                            })
                    
                    # Add to results
                    results.append({
                        "question_id": question_id,
                        "question": question_text,
                        "answer": answer,
                        "chunks": retrieved_chunks,
                        "evidences": evidences,
                        "category": category,
                        "conv_id": conv_id
                    })
                    
                    processed_questions += 1
                    
                except Exception as e:
                    print(f"\n  Error retrieving for question {question_id}: {e}")
                    continue
        
        except Exception as e:
            print(f"\n❌ Error loading memory system for {conv_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save retrieval results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Retrieval completed!")
    print(f"   Total questions: {total_questions}")
    print(f"   Processed questions: {processed_questions}")
    print(f"   Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve memories for questions using A-mem")
    parser.add_argument("input_file", type=str, help="Input JSON file with conversations and questions")
    parser.add_argument("memory_dir", type=str, help="Directory containing indexed memory systems")
    parser.add_argument("output_file", type=str, help="Output file for retrieval results")
    parser.add_argument("--top_k", type=int, default=100, 
                        help="Number of memories to retrieve per question")
    
    args = parser.parse_args()
    
    process_retrieval(
        input_file=args.input_file,
        memory_dir=args.memory_dir,
        output_file=args.output_file,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

