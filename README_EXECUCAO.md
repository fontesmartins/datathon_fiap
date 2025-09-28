# 🚀 Decision Recruitment AI - Guia de Execução

## 📋 Resumo Executivo

Sistema completo de Machine Learning para match candidato-vaga usando XGBoost, MLflow e FastAPI. O projeto foi otimizado e limpo, mantendo apenas os arquivos essenciais para execução.

## 🎯 Execução Rápida

### 1. Pipeline Completo
```bash
python main.py
```
**Resultado:** Modelo treinado com AUC 83.30%, salvo em `models/` e registrado no MLflow.

### 2. API REST
```bash
python fastapi_app.py
```
**Resultado:** API disponível em http://localhost:8000 com documentação em `/docs`.

### 3. MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```
**Resultado:** Interface web em http://localhost:5000 para visualizar experimentos.

## 📊 Performance Atual

| Métrica | Valor |
|---------|-------|
| **AUC Score** | 83.30% |
| **CV AUC** | 77.51% |
| **Features** | 30 |
| **Tempo Treinamento** | ~2-3 min |
| **Tempo Predição** | ~50ms |

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   main.py       │───▶│   MLflow        │───▶│   models/       │
│   (Pipeline)    │    │   (Tracking)    │    │   (Saved)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ fastapi_app.py  │    │ monitoring.py   │    │ optimization_   │
│   (API REST)    │    │ (Monitoring)    │    │ results.json    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Estrutura Final

```
fiap-final/
├── 🎯 main.py                    # Pipeline principal
├── 🌐 fastapi_app.py            # API REST
├── ⚙️ mlflow_config.py          # Configuração MLflow
├── 🧪 mlflow_experiments.py     # Experimentos
├── 🔧 model_pipeline.py         # Pipeline avançado
├── 📊 monitoring.py             # Monitoramento
├── 📈 data_analysis.py          # Análise de dados
├── 🏠 app.py                    # App principal
├── 🚀 deploy.py                 # Deploy
├── 📦 requirements.txt          # Dependências
├── 📊 dataset_preparado.csv     # Dataset
├── 📁 models/                   # Modelos salvos
├── 📁 mlruns/                   # MLflow tracking
└── 📄 optimization_results.json # Parâmetros otimizados
```

## 🔄 Fluxo de Execução

### 1. **Preparação**
- Instalar dependências: `pip install -r requirements.txt`

### 2. **Treinamento**
- Executar: `python main.py`
- **O que acontece:**
  - Carrega dados (53,759 registros)
  - Aplica feature engineering (30 features)
  - Otimiza hiperparâmetros (30 trials)
  - Treina modelo XGBoost
  - Salva modelo em `models/`
  - Registra no MLflow

### 3. **API**
- Executar: `python fastapi_app.py`
- **O que acontece:**
  - Carrega modelo treinado
  - Inicia servidor REST na porta 8000
  - Disponibiliza endpoints de predição

### 4. **Monitoramento**
- Executar: `python monitoring.py`
- **O que acontece:**
  - Verifica saúde da API
  - Testa endpoints
  - Calcula métricas
  - Gera relatório

### 5. **Visualização**
- Executar: `mlflow ui --backend-store-uri file:./mlruns`
- **O que acontece:**
  - Inicia interface web
  - Mostra experimentos e métricas
  - Permite navegar pelo Model Registry

## 🌐 Endpoints da API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Status da aplicação |
| `/predict` | POST | Predição individual |
| `/predict_batch` | POST | Predição em lote |
| `/model_info` | GET | Informações do modelo |
| `/feature_importance` | GET | Importância das features |
| `/mlflow_info` | GET | Status do MLflow |

## 🧪 Exemplo de Uso

### Predição Individual
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "candidate": {
    "nome": "João Silva",
    "nivel_profissional_candidato": "Sênior",
    "nivel_ingles_candidato": "Avançado",
    "nivel_espanhol_candidato": "Intermediário",
    "cv_text": "Python, AWS, Docker",
    "pcd": "Não",
    "remuneracao": 8000.0,
    "estado": "São Paulo"
  },
  "job": {
    "titulo_vaga": "Desenvolvedor Python",
    "nivel_profissional_vaga": "Sênior",
    "nivel_ingles_vaga": "Avançado",
    "nivel_espanhol_vaga": "Básico",
    "vaga_sap": "Não",
    "competencia_tecnicas": "Python, Django, AWS",
    "cliente": "TechCorp",
    "tipo_contratacao": "CLT Full"
  }
}'
```

### Resposta
```json
{
  "prediction": 0,
  "probability": 0.1569,
  "confidence": "Low",
  "recommendation": "NÃO RECOMENDADO",
  "explanation": "Candidato não atende aos critérios ideais para a vaga",
  "timestamp": "2025-09-14T11:40:22.931352",
  "status": "success"
}
```

## 📊 MLflow

### Acessar Interface
- URL: http://localhost:5000
- **Experimentos:** Visualizar todos os runs
- **Métricas:** AUC, CV AUC, tempo de execução
- **Model Registry:** Modelos registrados
- **Artefatos:** Modelos, gráficos, relatórios

### Informações do Experimento
- **Nome:** decision-recruitment-ai
- **Runs:** 1 run principal
- **Melhor AUC:** 83.30%
- **Modelo:** decision-recruitment-model v1

## 🔍 Verificações

### ✅ Pipeline Executado
- [ ] `models/` criado com arquivos
- [ ] `mlruns/` criado com experimentos
- [ ] `optimization_results.json` gerado
- [ ] AUC > 80% alcançado

### ✅ API Funcionando
- [ ] http://localhost:8000/health retorna 200
- [ ] http://localhost:8000/docs acessível
- [ ] Predições funcionando
- [ ] Logs sendo gerados

### ✅ MLflow Ativo
- [ ] http://localhost:5000 acessível
- [ ] Experimento visível
- [ ] Modelo no Registry
- [ ] Métricas logadas

## 🚨 Troubleshooting

### Problemas Comuns

**Porta 8000 ocupada:**
```bash
# Matar processo na porta 8000
lsof -ti:8000 | xargs kill -9
```

**MLflow UI não inicia:**
```bash
# Tentar porta diferente
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

**Modelo não carrega:**
- Verificar se `models/` existe
- Verificar se `model_metadata.json` está presente
- Verificar logs da API

**Otimização falha:**
- Verificar se dataset existe
- Verificar memória disponível
- Reduzir `n_trials` se necessário

## 🎯 Próximos Passos

1. **Deploy:** Usar `deploy.py` para produção
2. **Monitoramento:** Configurar alertas automáticos
3. **Retreinamento:** Automatizar retreinamento periódico
4. **A/B Testing:** Implementar testes de modelos
5. **CI/CD:** Integrar com pipeline de CI/CD

## 📚 Documentação Completa

- **GUIA_EXECUCAO.md** - Documentação detalhada de cada módulo
- **FLUXO_EXECUCAO.md** - Diagramas e fluxos visuais
- **README.md** - Documentação geral do projeto

## 🏆 Resultados Finais

**O projeto está completamente funcional com:**
- ✅ Pipeline de treinamento otimizado
- ✅ API REST funcionando
- ✅ MLflow integrado e ativo
- ✅ Sistema de monitoramento
- ✅ Documentação completa
- ✅ Código limpo e organizado

**Performance alcançada:**
- 🎯 AUC Score: 83.30%
- ⚡ Tempo de predição: ~50ms
- 🔧 30 features otimizadas
- 📊 30 trials de otimização
- 🤖 Modelo registrado no MLflow
