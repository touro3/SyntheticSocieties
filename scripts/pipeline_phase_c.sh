#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ==============================================================================
# PIPELINE FASE C: Baselines e Validação do Framework
# ==============================================================================

set -e

echo "======================================================"
echo " INICIANDO PIPELINE FASE C (COMPARAÇÃO DE BASELINES)"
echo "======================================================"

source venv/bin/activate
export PYTHONPATH="."

echo ""
echo "[1/2] Executando simulações de Baseline (Random, Template, Rule-Based)..."
# Supondo que você tenha um script unificado para rodar os baselines
# python scripts/run_baselines.py 

echo ""
echo "[2/2] Gerando gráficos de Macroeconomia e Distribuição (Gini/Cooperação)..."
python scripts/plot_phase_c.py

echo ""
echo "✅ PIPELINE DA FASE C CONCLUÍDO!"
echo "======================================================"