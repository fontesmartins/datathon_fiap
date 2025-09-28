#!/bin/bash

# Script de Teste da API Decision Recruitment AI
# Uso: ./test_api.sh [URL_BASE]
# Exemplo: ./test_api.sh http://localhost:8000
# Exemplo para nuvem: ./test_api.sh https://sua-api.herokuapp.com

# URL base da API (padrão: localhost)
API_URL=${1:-"http://localhost:8000"}

echo " Testando API Decision Recruitment AI"
echo " URL Base: $API_URL"
echo "=========================================="

# Função para testar endpoint
test_endpoint() {
local method=$1
local endpoint=$2
local data_file=$3
local description=$4

echo ""
echo " Testando: $description"
echo " $method $endpoint"

if [ -n "$data_file" ] && [ -f "$data_file" ]; then
response=$(curl -s -X $method "$API_URL$endpoint" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d @$data_file)
else
response=$(curl -s -X $method "$API_URL$endpoint" \
-H "accept: application/json")
fi

# Verificar se a resposta contém "status"
if echo "$response" | grep -q '"status"'; then
echo " Sucesso"
echo " Resposta: $response"
else
echo " Erro ou resposta inesperada"
echo " Resposta: $response"
fi
}

# Testes básicos
test_endpoint "GET" "/" "" "Endpoint raiz"
test_endpoint "GET" "/health" "" "Health check"
test_endpoint "GET" "/model_info" "" "Informações do modelo"
test_endpoint "GET" "/feature_importance" "" "Importância das features"
test_endpoint "GET" "/mlflow_info" "" "Informações do MLflow"

# Testes de predição
if [ -f "test_prediction.json" ]; then
test_endpoint "POST" "/predict" "test_prediction.json" "Predição individual - Desenvolvedor Python"
fi

if [ -f "test_prediction_positive.json" ]; then
test_endpoint "POST" "/predict" "test_prediction_positive.json" "Predição individual - Analista SAP"
fi

if [ -f "test_batch_prediction.json" ]; then
test_endpoint "POST" "/predict_batch" "test_batch_prediction.json" "Predição em lote"
fi

echo ""
echo "=========================================="
echo " Testes concluídos!"
echo " Documentação disponível em: $API_URL/docs"
echo " ReDoc disponível em: $API_URL/redoc"
