#!/bin/bash
set -e

echo "🔥 INICIANDO REPRODUÇÃO TOTAL DO ARTIGO BGF 🔥"

# Chama os outros scripts em sequência
bash pipeline_phase_c.sh
bash pipeline_phase_d.sh

echo "🏆 TODAS AS SIMULAÇÕES E GRÁFICOS FORAM GERADOS COM SUCESSO!"