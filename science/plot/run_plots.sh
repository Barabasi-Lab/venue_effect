#!/bin/bash
# ==============================================================================
# Plot all matched files in ../../data/matches/
#
# Usage:
#   bash run_plots.sh
#   bash run_plots.sh ../../data/matches ../../figures
# ==============================================================================

MATCH_DIR=${1:-../../data/matches}
FIG_DIR=${2:-../../figures}
LOG_FILE="${FIG_DIR}/plot_log.csv"

echo "============================================"
echo "  Plotting all matched files"
echo "  Input:  ${MATCH_DIR}"
echo "  Output: ${FIG_DIR}"
echo "  Log:    ${LOG_FILE}"
echo "============================================"

# Clear old log
rm -f "${LOG_FILE}"

COUNT=0
for CSV in ${MATCH_DIR}/merged_*.csv; do
    [ -f "$CSV" ] || continue
    BASENAME=$(basename "$CSV")
    echo ""
    echo "--- ${BASENAME} ---"
    python plot_matched_general.py --file "$CSV" --save --output_dir "$FIG_DIR" --log_file "$LOG_FILE"
    COUNT=$((COUNT + 1))
done

echo ""
echo "============================================"
echo "  Done: ${COUNT} files plotted"
echo "  Figures: ${FIG_DIR}"
echo "  Log: ${LOG_FILE}"
echo "============================================"
