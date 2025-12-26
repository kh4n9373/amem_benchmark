cd "$(dirname "$0")"

echo "🚀 Running full benchmark for LoCoMo dataset with A-mem..."
echo ""

# Use timestamped directories to avoid permission issues
timestamp=$(date +%Y%m%d_%H%M%S)

python3 amem_full_pipeline.py \
    data/locomo/processed_data/locomo_lite.json \
    locomo_memory_benchmark \
    locomo_results_${timestamp} \
    --max_workers 2 \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --embedding_model facebook/contriever \
    --evo_threshold 100 \
    --top_k 100 \
    --context_k 5 \
    --eval_ks "3,5,10" \
    --disable_thinking \
    --disable_thinking \
    --log_dir worker_logs/locomo

echo ""
echo "✅ LoCoMo benchmark completed!"
echo "📂 Results saved to: locomo_results_${timestamp}/"

