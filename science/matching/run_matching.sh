#!/bin/bash
# ==============================================================================
# Run matching for all journals in a field
#
# Usage:
#   bash run_matching.sh physics          # all journals, 32 cores
#   bash run_matching.sh biology 16       # 16 cores
#   bash run_matching.sh physics 1 50     # sequential, test=50
# ==============================================================================

FIELD=${1:?Usage: bash run_matching.sh <field> [n_jobs] [test]}
N_JOBS=${2:-32}
TEST=${3:-0}
COUNTRY_MAP="../../data/matching_needed/country_region.csv"

echo "============================================"
echo "  Matching: ${FIELD} | n_jobs=${N_JOBS} | test=${TEST}"
echo "============================================"

declare -A PHYSICS=(
    ["jour.1018957"]="Nature" ["jour.1346339"]="Science" ["jour.1082971"]="PNAS"
    ["jour.1034717"]="Nature_Physics" ["jour.1018277"]="PRL"
    ["jour.1053349"]="PRA" ["jour.1320488"]="PRB"
    ["jour.1320490"]="PRC" ["jour.1320496"]="PRD" ["jour.1312290"]="PRE")

declare -A BIOLOGY=(
    ["jour.1018957"]="Nature" ["jour.1346339"]="Science" ["jour.1082971"]="PNAS"
    ["jour.1019114"]="Cell" ["jour.1103138"]="Nature_Genetics"
    ["jour.1021344"]="Nature_Cell_Bio" ["jour.1295033"]="Nature_Struct")

declare -A CHEMISTRY=(
    ["jour.1018957"]="Nature" ["jour.1346339"]="Science" ["jour.1082971"]="PNAS"
    ["jour.1081898"]="JACS" ["jour.1017044"]="Angewandte"
    ["jour.1041224"]="Nature_Chemistry" ["jour.1155085"]="Chem")

declare -A SOCIOLOGY=(
    ["jour.1430837"]="Ann_Rev_Soc" ["jour.1017026"]="ASR"
    ["jour.1027842"]="ESR" ["jour.1068714"]="Social_Forces"
    ["jour.1126009"]="Gender_Society" ["jour.1008496"]="Sociology")

case $FIELD in
    physics)   declare -n J=PHYSICS ;;
    biology)   declare -n J=BIOLOGY ;;
    chemistry) declare -n J=CHEMISTRY ;;
    sociology) declare -n J=SOCIOLOGY ;;
    *) echo "Unknown field: $FIELD"; exit 1 ;;
esac

for JID in "${!J[@]}"; do
    echo ""
    echo "--- ${J[$JID]} (${JID}) ---"
    CMD="python matching.py --field ${FIELD} --journal_id ${JID} --n_jobs ${N_JOBS} --country_map ${COUNTRY_MAP}"
    [ "$TEST" -gt 0 ] && CMD="${CMD} --test ${TEST}"
    echo "  ${CMD}"
    ${CMD}
    [ $? -ne 0 ] && echo "  ERROR for ${JID}"
done

echo ""
echo "============================================"
echo "  Done: ${FIELD}"
echo "============================================"