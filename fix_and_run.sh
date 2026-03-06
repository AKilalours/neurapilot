#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# NeuraPilot — Fix & Run Script for Mac
# Run this ONCE from the neurapilot folder:
#   cd /Users/akilalourdes/Downloads/neurapilot
#   bash fix_and_run.sh
# ─────────────────────────────────────────────────────────────────

echo "📦 NeuraPilot Setup & Launch"
echo "────────────────────────────"

# Step 1: Make sure we're in the right folder
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "✅ Working directory: $SCRIPT_DIR"

# Step 2: Install the package in editable mode
# This permanently fixes "No module named 'neurapilot'"
echo ""
echo "📦 Installing neurapilot package (fixes module error)..."
pip install -e . --quiet
if [ $? -eq 0 ]; then
    echo "✅ Package installed successfully"
else
    echo "⚠️  pip install failed, trying pip3..."
    pip3 install -e . --quiet
fi

# Step 3: Install viz dependencies
echo ""
echo "📦 Installing visualization dependencies..."
pip install matplotlib numpy graphviz --quiet
echo "✅ Viz dependencies installed"

# Step 4: Add the package dir to PYTHONPATH as backup
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/neurapilot:$PYTHONPATH"
echo "✅ PYTHONPATH set"

# Step 5: Launch Streamlit from the inner package dir
echo ""
echo "🚀 Launching NeuraPilot..."
echo "────────────────────────────"
cd "$SCRIPT_DIR/neurapilot"
streamlit run ui_streamlit.py
