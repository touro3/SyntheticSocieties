#!/bin/bash

# ==============================================================================
# PIPELINE FASE D: Simulação em Larga Escala e Análise Topológica
# Este script executa a simulação BGF com 500 agentes, exporta os grafos 
# e gera as visualizações de rede em alta resolução.
# ==============================================================================

# Trava de segurança: interrompe o script se qualquer comando falhar
set -e
set -o pipefail

echo "======================================================"
echo " INICIANDO PIPELINE FASE D (ESCALA MÁXIMA)"
echo "======================================================"

# 1. Ativa o ambiente virtual e configura o PYTHONPATH
source venv/bin/activate
export PYTHONPATH="."

# 2. Executar a Simulação (Inferência em Lote)
echo ""
echo "[1/3] Executando o Kernel de Simulação (500 Agentes)..."
# Captura o diretório de artefatos mais recente automaticamente
RUN_DIR=$(ls -td artifacts/persona_fidelity/persona_fidelity_* | head -n 1)

python scripts/run_phase_d_scaling.py \
    --artifact-dir "$RUN_DIR" \
    --pop-size 500 \
    --rounds 30

# 3. Exportar Topologia de Redes (Gephi)
echo ""
echo "[2/3] Extraindo topologia e gerando arquivos .gexf..."
python scripts/export_gephi.py

# 4. Renderizar Visualizações Avançadas
echo ""
echo "[3/3] Renderizando gráficos de rede em alta resolução (400 DPI)..."
python scripts/plot_networks.py

echo ""
echo "✅ PIPELINE DA FASE D CONCLUÍDO COM SUCESSO!"
echo "📍 Visualizações salvas em: analysis/figures/"
echo "======================================================"