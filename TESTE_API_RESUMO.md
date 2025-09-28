# Resumo dos Testes da API - Decision Recruitment AI

## Status dos Testes 

Todos os endpoints da API foram testados com sucesso! A API está funcionando perfeitamente.

## Endpoints Testados

### 1. Health Check 
```bash
curl -X GET "http://localhost:8000/health"
```
**Resposta:** `{"status":"healthy","timestamp":"2025-09-27T23:33:18.625130","model_loaded":true}`

### 2. Endpoint Raiz 
```bash
curl -X GET "http://localhost:8000/"
```
**Resposta:** `{"message":"Decision Recruitment AI API","version":"1.0.0","docs":"/docs"}`

### 3. Informações do Modelo 
```bash
curl -X GET "http://localhost:8000/model_info"
```
**Resposta:** Modelo XGBoost com 30 features, treinado em 2025-09-14

### 4. Importância das Features 
```bash
curl -X GET "http://localhost:8000/feature_importance"
```
**Resposta:** Top 20 features mais importantes, lideradas por "is_sp" (0.147)

### 5. Predição Individual 
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @test_prediction.json
```
**Resposta:** Predição com probabilidade e recomendação

### 6. Predição em Lote 
```bash
curl -X POST "http://localhost:8000/predict_batch" -H "Content-Type: application/json" -d @test_batch_prediction.json
```
**Resposta:** Predições para múltiplos candidatos

### 7. Informações do MLflow 
```bash
curl -X GET "http://localhost:8000/mlflow_info"
```
**Resposta:** Informações do experimento e runs recentes

### 8. Documentação Interativa 
```bash
curl -X GET "http://localhost:8000/docs"
```
**Resposta:** Interface Swagger UI disponível

## Arquivos de Teste Criados

1. **test_prediction.json** - Exemplo de candidato desenvolvedor Python
2. **test_prediction_positive.json** - Exemplo de candidato analista SAP
3. **test_batch_prediction.json** - Exemplo de predição em lote com 2 candidatos

## Como Executar os Testes

### 1. Iniciar a API
```bash
cd /Users/kielmartins/Desktop/code/dev/fiap-final
python fastapi_app.py
```

### 2. Testar Endpoints
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Informações do modelo
curl -X GET "http://localhost:8000/model_info"

# Predição individual
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @test_prediction.json

# Predição em lote
curl -X POST "http://localhost:8000/predict_batch" -H "Content-Type: application/json" -d @test_batch_prediction.json
```

### 3. Acessar Documentação
Abra no navegador: `http://localhost:8000/docs`

## Observações Importantes

- Modelo carregado com sucesso (30 features)
- MLflow configurado e funcionando
- Todos os endpoints respondendo corretamente
- Logs de predições sendo registrados no MLflow
- Documentação automática disponível

## Para Teste na Nuvem

1. Use os arquivos JSON de exemplo fornecidos
2. Substitua `localhost:8000` pela URL da sua API na nuvem
3. Todos os endpoints estão prontos para produção
4. A API suporta tanto predições individuais quanto em lote

## Estrutura da API

- **FastAPI** com documentação automática
- **Pydantic** para validação de dados
- **MLflow** para tracking de predições
- **XGBoost** como modelo de ML
- **Logging** completo de todas as operações
