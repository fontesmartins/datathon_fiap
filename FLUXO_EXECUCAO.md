# 🔄 Fluxo de Execução - Decision Recruitment AI

## 📊 Diagrama do Fluxo

```
┌─────────────────────────────────────────────────────────────────┐
│                    🚀 PIPELINE COMPLETO                        │
└─────────────────────────────────────────────────────────────────┘

1️⃣ INICIALIZAÇÃO
   ┌─────────────────┐
   │   main.py       │
   │   __init__()    │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ mlflow_config.py│
   │ setup_mlflow()  │
   └─────────────────┘

2️⃣ CARREGAMENTO DE DADOS
   ┌─────────────────┐
   │   main.py       │
   │load_and_prepare │
   │_data()          │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ dataset_preparado│
   │     .csv        │
   └─────────────────┘

3️⃣ FEATURE ENGINEERING
   ┌─────────────────┐
   │   main.py       │
   │basic_feature_   │
   │engineering()    │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   main.py       │
   │prepare_features │
   │()               │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   main.py       │
   │encode_categorical│
   │_features()      │
   └─────────────────┘

4️⃣ OTIMIZAÇÃO DE HIPERPARÂMETROS
   ┌─────────────────┐
   │   main.py       │
   │optimize_hyper   │
   │parameters()     │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │     Optuna      │
   │  30 trials      │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │optimization_    │
   │results.json     │
   └─────────────────┘

5️⃣ TREINAMENTO DO MODELO
   ┌─────────────────┐
   │   main.py       │
   │train_final_     │
   │model()          │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   MLflow        │
   │   Logging       │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   XGBoost       │
   │   Model         │
   └─────────────────┘

6️⃣ SALVAMENTO E REGISTRO
   ┌─────────────────┐
   │   main.py       │
   │save_model()     │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   models/       │
   │   - xgboost_model.pkl│
   │   - label_encoders.pkl│
   │   - scaler.pkl  │
   │   - model_metadata.json│
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   main.py       │
   │register_model_  │
   │in_mlflow()      │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ MLflow Registry │
   │   Production    │
   └─────────────────┘

7️⃣ INICIALIZAÇÃO DA API
   ┌─────────────────┐
   │ fastapi_app.py  │
   │ startup_event() │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ fastapi_app.py  │
   │ load_model()    │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   API REST      │
   │   Port 8000     │
   └─────────────────┘

8️⃣ MONITORAMENTO
   ┌─────────────────┐
   │ monitoring.py   │
   │ main()          │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   Health Check  │
   │   API Tests     │
   │   Metrics       │
   └─────────────────┘

9️⃣ VISUALIZAÇÃO
   ┌─────────────────┐
   │   MLflow UI     │
   │   Port 5000     │
   └─────────────────┘
```

## 🔄 Fluxo de Predição (API)

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌐 FLUXO DE PREDIÇÃO                        │
└─────────────────────────────────────────────────────────────────┘

1️⃣ REQUISIÇÃO
   ┌─────────────────┐
   │   Cliente       │
   │   POST /predict │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ fastapi_app.py  │
   │ predict()       │
   └─────────────────┘

2️⃣ PREPROCESSAMENTO
   ┌─────────────────┐
   │ fastapi_app.py  │
   │preprocess_input │
   │()               │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ fastapi_app.py  │
   │apply_basic_     │
   │feature_         │
   │engineering()    │
   └─────────────────┘

3️⃣ PREDIÇÃO
   ┌─────────────────┐
   │   XGBoost       │
   │   Model         │
   │   .predict()    │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   Resultado     │
   │   + Probabilidade│
   │   + Confiança   │
   └─────────────────┘

4️⃣ LOGGING
   ┌─────────────────┐
   │   MLflow        │
   │   Logging       │
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   Resposta      │
   │   JSON          │
   └─────────────────┘
```

## 📁 Estrutura de Arquivos

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
│   ├── xgboost_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── 📁 mlruns/                   # MLflow tracking
└── 📄 optimization_results.json # Parâmetros otimizados
```

## 🎯 Comandos de Execução

### 1. Pipeline Completo
```bash
python main.py
```

### 2. API REST
```bash
python fastapi_app.py
```

### 3. Monitoramento
```bash
python monitoring.py
```

### 4. MLflow UI
```bash
mlflow ui --backend-store-uri file:./mlruns
```

### 5. Teste da API
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

## 📊 Métricas de Performance

| Etapa | Métrica | Valor |
|-------|---------|-------|
| **Otimização** | Melhor AUC | 83.30% |
| **Treinamento** | CV AUC | 77.51% |
| **Features** | Quantidade | 30 |
| **Trials** | Otimização | 30 |
| **Tempo** | Treinamento | ~2-3 min |
| **API** | Tempo resposta | ~50ms |

## 🔍 Pontos de Verificação

### ✅ Pipeline Executado
- [ ] Dados carregados (53,759 registros)
- [ ] Feature engineering aplicado
- [ ] Otimização concluída (30 trials)
- [ ] Modelo treinado (AUC > 80%)
- [ ] Modelo salvo em `models/`
- [ ] Modelo registrado no MLflow

### ✅ API Funcionando
- [ ] API iniciada na porta 8000
- [ ] Modelo carregado com sucesso
- [ ] Endpoint `/health` respondendo
- [ ] Endpoint `/predict` funcionando
- [ ] Predições sendo logadas no MLflow

### ✅ MLflow Ativo
- [ ] Diretório `mlruns/` criado
- [ ] Experimento registrado
- [ ] Runs logados com métricas
- [ ] Modelo no Model Registry
- [ ] UI acessível em http://localhost:5000

### ✅ Monitoramento
- [ ] Sistema de logs funcionando
- [ ] Métricas sendo calculadas
- [ ] Health checks passando
- [ ] Relatórios sendo gerados
