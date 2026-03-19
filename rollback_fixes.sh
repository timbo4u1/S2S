#!/bin/bash
# S2S Rollback Script - Restore from backup

set -e

echo "════════════════════════════════════════════════════════════════"
echo "S2S ROLLBACK - RESTORE FROM BACKUP"
echo "════════════════════════════════════════════════════════════════"
echo ""

S2S_DIR="$HOME/S2S"
BACKUP_BASE="$S2S_DIR/backups"

# Check if backup directory exists
if [ ! -d "$BACKUP_BASE" ]; then
    echo "❌ No backups found in $BACKUP_BASE"
    exit 1
fi

# List available backups
echo "Available backups:"
echo ""
ls -1dt "$BACKUP_BASE"/* | head -5 | nl

echo ""
read -p "Enter backup number to restore (or 'q' to quit): " choice

if [ "$choice" = "q" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Get the selected backup
BACKUP_DIR=$(ls -1dt "$BACKUP_BASE"/* | head -5 | sed -n "${choice}p")

if [ -z "$BACKUP_DIR" ]; then
    echo "❌ Invalid selection"
    exit 1
fi

echo ""
echo "Restoring from: $BACKUP_DIR"
echo ""
read -p "Are you sure? This will overwrite current files. (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Restore files
echo ""
echo "Restoring files..."

if [ -f "$BACKUP_DIR/s2s_standard_v1_3/s2s_registry_v1_3.py" ]; then
    cp "$BACKUP_DIR/s2s_standard_v1_3/s2s_registry_v1_3.py" "$S2S_DIR/s2s_standard_v1_3/"
    echo "  ✅ Restored: s2s_registry_v1_3.py"
fi

if [ -f "$BACKUP_DIR/s2s_standard_v1_3/s2s_fusion_v1_3.py" ]; then
    cp "$BACKUP_DIR/s2s_standard_v1_3/s2s_fusion_v1_3.py" "$S2S_DIR/s2s_standard_v1_3/"
    echo "  ✅ Restored: s2s_fusion_v1_3.py"
fi

if [ -f "$BACKUP_DIR/experiments/step3_retrieval.py" ]; then
    cp "$BACKUP_DIR/experiments/step3_retrieval.py" "$S2S_DIR/experiments/"
    echo "  ✅ Restored: step3_retrieval.py"
fi

echo ""
echo "✅ Rollback complete! Original files restored."
echo ""
