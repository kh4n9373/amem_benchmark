#!/bin/bash
# Setup script for A-mem benchmark pipeline (FIXED: uv editable git error + SQLite)

set -e

echo "🚀 A-mem Benchmark Setup (uv + SQLite fix)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. Kiểm tra uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv chưa được cài. Đang cài đặt..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 1. Tạo môi trường ảo
echo "📌 Tạo virtual environment (Python 3.10)..."
uv venv .venv --python 3.10
source .venv/bin/activate
echo "✅ Đã kích hoạt .venv"

# 2. Setup TMPDIR
export TMPDIR="./.tmp"
mkdir -p "$TMPDIR"

# 3. Cài đặt dependencies
echo "📌 Đang cài dependencies..."

# --- [BƯỚC 1] Cài gói fix lỗi SQLite ---
echo "   -> Cài đặt pysqlite3-binary (Fix lỗi ChromaDB)..."
uv pip install pysqlite3-binary

# --- [BƯỚC 2] Xử lý requirements.txt (Lọc bỏ dòng -e git+ gây lỗi) ---
if [ -f "requirements.txt" ]; then
    echo "   -> Đang xử lý requirements.txt..."
    # Tạo file tạm, loại bỏ dòng chứa 'git+' và '-e' đi cùng nhau
    grep -vE "^\s*-e\s+git\+" requirements.txt > requirements.tmp
    
    echo "   -> Cài đặt từ file đã lọc..."
    uv pip install -r requirements.tmp
    rm requirements.tmp # Xóa file tạm
else
    echo "⚠️ Không thấy requirements.txt"
fi

# --- [BƯỚC 3] Cài package A-mem từ local (Thay thế cho dòng git vừa xóa) ---
if [ -d "amem" ]; then
    echo "   -> Cài package A-mem (Local Editable)..."
    uv pip install -e "amem"
fi

# 4. [QUAN TRỌNG] Tự động sửa code để nhận SQLite mới (Cách cũ - Backup)
echo "📌 Đang patch code (Backup method)..."
TARGET_FILE="amem/agentic_memory/memory_system.py"
if [ -f "$TARGET_FILE" ]; then
    if ! grep -q "sys.modules\['sqlite3'\]" "$TARGET_FILE"; then
        sed -i "1s|^|__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')\n|" "$TARGET_FILE"
        echo "✅ Đã patch file memory_system.py"
    fi
fi

# 5. [QUAN TRỌNG NHẤT] Fix toàn cục bằng sitecustomize (Chữa tận gốc)
echo "📌 Đang tiêm thuốc fix SQLite vào hệ thống (Sitecustomize)..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cat <<EOF > "$SITE_PACKAGES/sitecustomize.py"
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
EOF
echo "✅ Đã tạo file sitecustomize.py tại $SITE_PACKAGES"


# 6. Download NLTK & Data
echo "📌 Kiểm tra dữ liệu..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

if [ ! -f "data/locomo/processed_data/locomo_processed_data.json" ]; then
    echo "▶ Downloading dataset..."
    mkdir -p data
    python <<'EOF'
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id="KhangPTT373/locomo", local_dir="data/locomo", repo_type="dataset")
except Exception as e: exit(1)
EOF
fi

mkdir -p worker_logs/locomo

# 7. Test
echo "📌 Test thử import..."
python -c "import sqlite3; print(f'🔥 SQLite version đang dùng: {sqlite3.sqlite_version}'); from agentic_memory.memory_system import AgenticMemorySystem; print('✅ A-mem import OK!')"

echo ""
echo "=========================================="
echo "✅ Cài đặt hoàn tất!"
echo "⚠️  QUAN TRỌNG: Trước khi chạy lệnh khác, hãy gõ:"
echo "   source .venv/bin/activate"
echo "=========================================="