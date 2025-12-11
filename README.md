# A-mem Benchmark Pipeline

Complete benchmark pipeline for A-mem memory system on conversational datasets (LoCoMo and LongMemEval).

## Features

- **Full A-mem Integration**: Uses all A-mem features including LLM-based keyword extraction, context generation, auto-linking, and memory evolution
- **Per-Turn Chunking**: Groups user+assistant message pairs as coherent memory units
- **Parallel Processing**: Multi-worker indexing for faster processing
- **Comprehensive Evaluation**: Both retrieval and generation metrics
- **Easy Setup**: One-script setup with conda environment

## Quick Start

### 1. Setup

```bash
# Run setup script (installs dependencies, A-mem, datasets)
bash setup.sh

# Activate conda environment
conda activate amem
```

### 2. Quick Test (20 conversations)

```bash
# Test the pipeline on 20 conversations (~10-15 minutes)
bash quick_test.sh
```

### 3. Full Benchmark

```bash
# Run full benchmark on LoCoMo dataset
bash full_benchmark_locomo.sh

# Or run on LongMemEval dataset
bash full_benchmark_longmemeval.sh
```

## Architecture

```
Dataset (LoCoMo/LongMemEval)
  ↓
Per-Turn Chunking (user + assistant)
  ↓
Index with A-mem (LLM extraction, linking, evolution)
  ↓
Store in ChromaDB
  ↓
Retrieve with search_agentic()
  ↓
Evaluate (Retrieval + Generation metrics)
```

## Pipeline Steps

1. **Indexing** (`amem_process_index.py`)
   - Load conversational data
   - Extract turn pairs (user + assistant)
   - Index into A-mem with full features
   - LLM automatically extracts keywords, context, tags
   - Auto-linking of similar memories
   - Periodic consolidation (every N notes)

2. **Retrieval** (`amem_process_retrieve.py`)
   - Load indexed memory systems
   - Use `search_agentic()` to retrieve top-K memories
   - Include rich metadata (keywords, tags, context, timestamp)

3. **Evaluation**
   - **Retrieval Metrics**: Precision@K, Recall@K, F1@K, nDCG@K
   - **Generation Metrics**: F1, BLEU, ROUGE, BERTScore

## Configuration

### Key Parameters

- `--max_workers`: Number of parallel indexing workers (default: 2)
- `--llm_model`: LLM model for A-mem (default: Qwen/Qwen3-8B)
- `--embedding_model`: Embedding model for ChromaDB (default: all-MiniLM-L6-v2)
- `--evo_threshold`: Consolidate memories every N notes (default: 100)
- `--top_k`: Number of memories to retrieve (default: 100)
- `--context_k`: Memories to use as context for generation (default: 5)
- `--eval_ks`: K values for evaluation (default: "3,5,10")

### Example Usage

```bash
python3 amem_full_pipeline.py \
    data/locomo/processed_data/locomo_processed_data.json \
    output/memory \
    output/results \
    --max_workers 2 \
    --llm_model Qwen/Qwen3-8B \
    --embedding_model all-MiniLM-L6-v2 \
    --evo_threshold 100 \
    --top_k 100
```

## Output Structure

```
output_dir/
├── retrieval_results_TIMESTAMP.json    # Retrieved memories with scores
├── retrieval_eval_TIMESTAMP.json       # Retrieval metrics (P, R, F1, nDCG)
├── generation_eval_TIMESTAMP.json      # Generation metrics (BLEU, ROUGE, BERTScore)
└── pipeline_metadata_TIMESTAMP.json    # Config and timing info

memory_dir/
├── conv-1/
│   └── memory_system.pkl               # Pickled A-mem system
├── conv-2/
│   └── memory_system.pkl
└── index_metadata_0.json               # Indexing metadata
```

## A-mem vs Mem0

| Aspect | Mem0 | A-mem |
|--------|------|-------|
| **Chunking** | Per-message, sliding window | Per-turn (user+assistant) |
| **LLM Usage** | Optional (`infer=True/False`) | Always enabled |
| **Metadata** | Simple dict | Rich (keywords, tags, context, links) |
| **Evolution** | Manual | Automatic consolidation |
| **Linking** | Manual | Automatic similarity-based |
| **Retrieval** | `memory.search()` | `search_agentic()` with neighbor expansion |

## Performance Notes

- **Indexing Speed**: ~5-10s per turn with LLM extraction
- **Memory Usage**: Keeps all MemoryNote objects in RAM
- **Evolution**: Consolidation happens every `evo_threshold` notes (slower but maintains quality)
- **Recommended**: Start with `--max_workers 2` and `--evo_threshold 100`

## Troubleshooting

### Import Errors

```bash
# Make sure A-mem is installed
pip install -e /home/vinhpq/mem_baseline/A-mem

# Check installation
python3 -c "from agentic_memory.memory_system import AgenticMemorySystem; print('OK')"
```

### Slow Indexing

- Increase `--evo_threshold` to reduce consolidation frequency
- Use more workers: `--max_workers 4`
- Check LLM server performance

### Out of Memory

- Reduce `--max_workers` to 1
- Process dataset in batches
- Increase system RAM

## References

- [A-mem Paper](https://arxiv.org/pdf/2502.12110)
- [A-mem GitHub](https://github.com/agiresearch/A-mem)
- [LoCoMo Dataset](https://huggingface.co/datasets/TeddyKT/LoCoMo)


