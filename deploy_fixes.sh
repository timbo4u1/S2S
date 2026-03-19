#!/bin/bash
# S2S Bug Fix Deployment Script
# Backs up original files, deploys fixes, runs tests

set -e  # Exit on any error

echo "════════════════════════════════════════════════════════════════"
echo "S2S BUG FIX DEPLOYMENT"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
S2S_DIR="$HOME/S2S"
BACKUP_DIR="$S2S_DIR/backups/$(date +%Y%m%d_%H%M%S)"
FIXED_DIR="$(dirname "$0")"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if S2S directory exists
if [ ! -d "$S2S_DIR" ]; then
    echo -e "${RED}❌ Error: S2S directory not found: $S2S_DIR${NC}"
    echo "Please update S2S_DIR in this script to point to your S2S installation"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  S2S Directory: $S2S_DIR"
echo "  Backup Directory: $BACKUP_DIR"
echo "  Fixed Files Directory: $FIXED_DIR"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Create Backup
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${YELLOW}[STEP 1] Creating backup...${NC}"

mkdir -p "$BACKUP_DIR"

# Files to backup
FILES_TO_BACKUP=(
    "s2s_standard_v1_3/s2s_registry_v1_3.py"
    "s2s_standard_v1_3/s2s_fusion_v1_3.py"
    "experiments/step3_retrieval.py"
)

for file in "${FILES_TO_BACKUP[@]}"; do
    source_file="$S2S_DIR/$file"
    if [ -f "$source_file" ]; then
        backup_file="$BACKUP_DIR/$file"
        mkdir -p "$(dirname "$backup_file")"
        cp "$source_file" "$backup_file"
        echo "  ✅ Backed up: $file"
    else
        echo -e "  ${YELLOW}⚠️  Not found (skipping): $file${NC}"
    fi
done

echo -e "${GREEN}✅ Backup complete: $BACKUP_DIR${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Run Tests on Fixed Versions
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${YELLOW}[STEP 2] Running tests on fixed versions...${NC}"

cd "$FIXED_DIR"
python3 test_all_fixes.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Tests failed! Not deploying.${NC}"
    echo "Review test output and fix issues before deployment."
    exit 1
fi

echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Deploy Fixed Files
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${YELLOW}[STEP 3] Deploying fixed files...${NC}"

# Deploy registry fix
if [ -f "$FIXED_DIR/s2s_registry_v1_3_FIXED.py" ]; then
    cp "$FIXED_DIR/s2s_registry_v1_3_FIXED.py" "$S2S_DIR/s2s_standard_v1_3/s2s_registry_v1_3.py"
    echo "  ✅ Deployed: s2s_registry_v1_3.py (security fix)"
else
    echo -e "  ${RED}❌ Fixed file not found: s2s_registry_v1_3_FIXED.py${NC}"
fi

# Deploy fusion fix
if [ -f "$FIXED_DIR/s2s_fusion_v1_3_FIXED.py" ]; then
    cp "$FIXED_DIR/s2s_fusion_v1_3_FIXED.py" "$S2S_DIR/s2s_standard_v1_3/s2s_fusion_v1_3.py"
    echo "  ✅ Deployed: s2s_fusion_v1_3.py (performance fix)"
else
    echo -e "  ${RED}❌ Fixed file not found: s2s_fusion_v1_3_FIXED.py${NC}"
fi

# Deploy retrieval fix
if [ -f "$FIXED_DIR/step3_retrieval_FIXED.py" ]; then
    cp "$FIXED_DIR/step3_retrieval_FIXED.py" "$S2S_DIR/experiments/step3_retrieval.py"
    echo "  ✅ Deployed: step3_retrieval.py (semantic fix)"
else
    echo -e "  ${RED}❌ Fixed file not found: step3_retrieval_FIXED.py${NC}"
fi

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${YELLOW}[STEP 4] Installing dependencies...${NC}"

# Check if sentence-transformers is installed
python3 -c "import sentence_transformers" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  📦 Installing sentence-transformers..."
    pip3 install sentence-transformers --break-system-packages || pip3 install sentence-transformers --user
    echo "  ✅ sentence-transformers installed"
else
    echo "  ✅ sentence-transformers already installed"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Verify Installation
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${YELLOW}[STEP 5] Verifying installation...${NC}"

cd "$S2S_DIR"

# Quick import test
python3 -c "
from s2s_standard_v1_3.s2s_registry_v1_3 import DeviceRegistry
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier
print('  ✅ All modules import correctly')
" || {
    echo -e "${RED}❌ Import test failed!${NC}"
    echo "Fixes may not be installed correctly."
    exit 1
}

echo -e "${GREEN}✅ Installation verified!${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo "DEPLOYMENT SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ All fixes deployed successfully!${NC}"
echo ""
echo "Fixed bugs:"
echo "  1. Security: Signature bypass vulnerability (s2s_registry_v1_3.py)"
echo "  2. Performance: O(n²) → O(n) fusion scaling (s2s_fusion_v1_3.py)"
echo "  3. Semantic: Physics encoder → Semantic encoder (step3_retrieval.py)"
echo ""
echo "Backup location:"
echo "  $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Test with real data: cd $S2S_DIR && python3 test_pamap2.py"
echo "  2. Run WESAD benchmark for Kristof's email"
echo "  3. Commit and push to GitHub:"
echo "     cd $S2S_DIR"
echo "     git add ."
echo "     git commit -m 'Fix: Security, performance, and semantic bugs'"
echo "     git push origin main"
echo ""
echo "Rollback (if needed):"
echo "  bash rollback_fixes.sh"
echo ""
echo "════════════════════════════════════════════════════════════════"
