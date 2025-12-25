#!/bin/bash
# Setup script for A-mem benchmark pipeline
# This script sets up the environment, installs dependencies, and prepares data

set -e  # Exit on error

echo "🚀 A-mem Benchmark Pipeline Setup"
echo "=================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Python version
echo "📌 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✅ Python check passed"
echo ""

# 2. Create/activate conda environment (optional but recommended)
echo "📌 Checking for conda environment 'amem'..."
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^amem "; then
        echo "✅ Conda environment 'amem' already exists"
    else
        echo "Creating conda environment 'amem'..."
        conda create -n amem python=3.10 -y
        echo "✅ Conda environment created"
    fi
    echo ""
    echo "💡 To activate: conda activate amem"
else
    echo "⚠️  Conda not found. Using system Python."
    echo "💡 Consider using conda or venv for better dependency isolation."
fi
echo ""

# 3. Set TMPDIR to avoid disk space issues
echo "📌 Setting TMPDIR to avoid disk space issues..."
export TMPDIR="/home/vinhpq/.tmp"
mkdir -p "$TMPDIR"
echo "✅ TMPDIR set to $TMPDIR"
echo ""

# 4. Install A-mem package
echo "📌 Installing A-mem package..."
A_MEM_DIR="$SCRIPT_DIR/amem"
if [ -d "$A_MEM_DIR" ]; then
    pip install -e "$A_MEM_DIR"
    echo "✅ A-mem package installed"
else
    echo "❌ A-mem directory not found at $A_MEM_DIR"
    exit 1
fi
echo ""

# 5. Install requirements
echo "📌 Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi
echo ""

# 6. Download NLTK data
echo "📌 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true
echo "✅ NLTK data downloaded"
echo ""

# 7. Check data symlink
echo "📌 Checking data symlink..."
if [ -L "data" ]; then
    echo "✅ Data symlink already exists"
elif [ ! -d "../mem0/data" ]; then
    echo "❌ mem0/data directory not found. Please run mem0 setup first."
    exit 1
else
    ln -sf ../mem0/data data
    echo "✅ Data symlink created"
fi
echo ""

# 8. Verify datasets
echo "📌 Verifying datasets..."
LOCOMO_FILE="data/locomo/processed_data/locomo_processed_data.json"
LONGMEMEVAL_FILE="data/locomo/processed_data/longmemeval_processed_data.json"

if [ -f "$LOCOMO_FILE" ]; then
    echo "✅ LoCoMo dataset found"
else
    echo "⚠️  LoCoMo dataset not found at $LOCOMO_FILE"
    echo "   Run: cd ../mem0 && bash setup.sh"
fi

if [ -f "$LONGMEMEVAL_FILE" ]; then
    echo "✅ LongMemEval dataset found"
else
    echo "⚠️  LongMemEval dataset not found at $LONGMEMEVAL_FILE"
    echo "   This dataset may need to be downloaded separately"
fi
echo ""

# 9. Create worker_logs directory
echo "📌 Creating worker_logs directory..."
mkdir -p worker_logs
echo "✅ worker_logs directory created"
echo ""

# 10. Test A-mem installation
echo "📌 Testing A-mem installation..."
python3 -c "from agentic_memory.memory_system import AgenticMemorySystem; print('✅ A-mem import successful')"
if [ $? -ne 0 ]; then
    echo "❌ A-mem import failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ Setup completed successfully!"
echo "=========================================="

cd "$(dirname "$0")"

echo "🚀 Running full benchmark for Longmemeval dataset with A-mem..."
echo ""

# Use timestamped directories to avoid permission issues
timestamp=$(date +%Y%m%d_%H%M%S)

python3 amem_full_pipeline.py \
    data/locomo/processed_data/longmemeval_processed_data.json \
    longmemeval_memory_benchmark_${timestamp} \
    longmemeval_results_${timestamp} \
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
    --disable_thinking

echo ""
echo "✅ Longmemeval benchmark completed!"
echo "📂 Results saved to: locomo_results_${timestamp}/"

