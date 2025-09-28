# 🚀 Guia de Execução - Decision Recruitment AI

## 📋 Visão Geral

Este documento explica a ordem de execução e funcionamento de cada módulo do sistema Decision Recruitment AI, um sistema de match candidato-vaga usando Machine Learning com XGBoost e MLflow.

## 🎯 Ordem de Execução

### 1. **main.py** - Pipeline Principal
**Arquivo principal que executa todo o fluxo de treinamento**

#### Classe: `DecisionRecruitmentPipeline`

##### Métodos principais:

**`__init__(self)`**
- Inicializa a pipeline
- Configura MLflow, scaler, encoders e listas de features
- Cria instância do MLflow config

**`load_and_prepare_data(self)`**
- Carrega dataset de `dataset_preparado.csv`
- Aplica feature engineering básico
- Prepara features numéricas e categóricas
- Codifica variáveis categóricas
- Retorna X (features) e y (target)

**`basic_feature_engineering(self, df)`**
- Cria features de compatibilidade (nível profissional, idiomas)
- Gera features de texto (CV analysis)
- Calcula features básicas (SAP, PCD, localização)
- Cria features de match (compatibilidade exata)
- Retorna DataFrame com novas features

**`prepare_features(self, df)`**
- Define lista de features numéricas (16 features)
- Define lista de features categóricas (14 features)
- Filtra colunas que existem no dataset

**`encode_categorical_features(self, df)`**
- Aplica LabelEncoder em cada feature categórica
- Cria colunas `_encoded` para cada categoria
- Salva encoders para uso posterior

**`prepare_training_data(self, df)`**
- Filtra features válidas
- Remove linhas com target nulo
- Identifica variáveis contínuas
- Aplica normalização (StandardScaler)
- Retorna X e y prontos para treinamento

**`identify_continuous_variables(self, X)`**
- Identifica variáveis que têm valores decimais
- Retorna lista de variáveis contínuas para normalização

**`optimize_hyperparameters(self, X, y, n_trials=50)`**
- Executa otimização bayesiana com Optuna
- Testa 30 combinações de hiperparâmetros
- Usa validação holdout (80% treino, 20% validação)
- Salva melhores parâmetros em `optimization_results.json`
- Retorna parâmetros otimizados e melhor score

**`train_final_model(self, X, y, best_params)`**
- Treina modelo final com parâmetros otimizados
- Usa MLflow para logging de parâmetros e métricas
- Executa validação cruzada (5-fold)
- Loga importância das features
- Registra modelo no MLflow
- Retorna métricas de performance

**`save_model(self)`**
- Salva modelo XGBoost em `models/xgboost_model.pkl`
- Salva encoders em `models/label_encoders.pkl`
- Salva scaler em `models/scaler.pkl`
- Salva metadados em `models/model_metadata.json`

**`register_model_in_mlflow(self)`**
- Registra modelo no MLflow Model Registry
- Define stage como "Production"
- Retorna status do registro

**`main()`**
- Função principal que orquestra todo o pipeline
- Executa: carregamento → otimização → treinamento → salvamento → registro
- Retorna pipeline e resultados

---

### 2. **mlflow_config.py** - Configuração MLflow
**Gerencia configuração e operações do MLflow**

#### Classe: `MLflowConfig`

##### Métodos principais:

**`__init__(self, experiment_name, tracking_uri, registry_uri)`**
- Configura tracking URI (file:./mlruns)
- Cria ou obtém experimento
- Define experimento ativo

**`setup_mlflow(self)`**
- Configura tracking e registry URIs
- Cria experimento se não existir
- Define experimento ativo

**`start_run(self, run_name, tags)`**
- Inicia novo run do MLflow
- Adiciona tags padrão (project, model_type, created_at)
- Retorna contexto do run

**`log_model_params(self, params)`**
- Loga parâmetros do modelo no MLflow
- Registra quantidade de parâmetros

**`log_model_metrics(self, metrics)`**
- Loga métricas do modelo
- Registra lista de métricas logadas

**`log_model_artifacts(self, artifacts)`**
- Loga arquivos como artefatos
- Verifica existência dos arquivos

**`log_xgboost_model(self, model, model_name, signature, input_example)`**
- Loga modelo XGBoost no MLflow
- Registra modelo com nome "decision-recruitment-model"
- Inclui signature e input example

**`log_feature_importance(self, feature_names, importance_scores)`**
- Cria DataFrame com importância das features
- Loga como artefato CSV
- Loga top 10 features como métricas

**`log_dataset_info(self, dataset_path, target_column)`**
- Loga informações do dataset
- Calcula métricas: linhas, colunas, distribuição target, valores nulos
- Loga dataset como artefato

**`register_model(self, model_name, model_version, stage)`**
- Registra modelo no Model Registry
- Transiciona para stage especificado (Production)
- Retorna modelo registrado

**`load_model(self, model_name, stage)`**
- Carrega modelo do Model Registry
- Usa URI: models:/{model_name}/{stage}
- Retorna modelo carregado

---

### 3. **fastapi_app.py** - API REST
**Interface REST para predições do modelo**

#### Endpoints principais:

**`load_model()`**
- Carrega modelo na inicialização da API
- Tenta carregar do MLflow Registry primeiro
- Fallback para modelo local se necessário
- Configura MLflow para logging de predições

**`preprocess_input(candidate_data, job_data)`**
- Preprocessa dados de entrada da API
- Aplica feature engineering básico
- Cria DataFrame com features necessárias
- Preenche features ausentes com 0

**`apply_basic_feature_engineering(df)`**
- Aplica mesmo feature engineering do treinamento
- Calcula compatibilidades (nível, idiomas)
- Analisa texto do CV
- Cria features de match

**`@app.on_event("startup")`**
- Evento de inicialização da aplicação
- Carrega modelo automaticamente
- Falha se modelo não puder ser carregado

**`@app.get("/health")`**
- Endpoint de health check
- Retorna status da aplicação e modelo

**`@app.post("/predict")`**
- Endpoint principal de predição
- Recebe dados do candidato e vaga
- Preprocessa dados
- Faz predição com modelo
- Determina confiança e recomendação
- Loga predição no MLflow
- Retorna resposta estruturada

**`@app.post("/predict_batch")`**
- Endpoint para predição em lote
- Processa múltiplos candidatos para uma vaga
- Retorna lista de resultados

**`@app.get("/model_info")`**
- Retorna informações do modelo
- Carrega metadados do arquivo JSON
- Lista features e configurações

**`@app.get("/feature_importance")`**
- Retorna importância das features
- Ordena por importância decrescente
- Retorna top 20 features

**`@app.get("/mlflow_info")`**
- Retorna informações do MLflow
- Lista runs recentes
- Mostra métricas dos experimentos

---

### 4. **monitoring.py** - Sistema de Monitoramento
**Monitora performance e saúde do sistema**

#### Classes principais:

**`ModelMonitor`**
- Monitor de predições e métricas
- Detecta drift no modelo
- Gera relatórios de performance

**`MLflowMonitor`**
- Monitor específico do MLflow
- Obtém métricas dos experimentos
- Monitora Model Registry

**`APIMonitor`**
- Monitor da API REST
- Testa endpoints
- Verifica saúde da aplicação

##### Métodos principais:

**`log_prediction(prediction_data, response_time_ms, model_version)`**
- Registra predição no log estruturado
- Salva timestamp, candidato, resultado, tempo de resposta

**`calculate_metrics(time_window_hours)`**
- Calcula métricas agregadas
- Taxa de recomendação, confiança, tempo de resposta
- Retorna objeto ModelMetrics

**`detect_drift(baseline_metrics, current_metrics, threshold)`**
- Detecta drift comparando métricas
- Compara taxa de recomendação, probabilidade média
- Retorna detalhes do drift detectado

**`generate_report(time_window_hours)`**
- Gera relatório completo de monitoramento
- Inclui métricas, status do sistema, recomendações

**`get_experiment_metrics()`**
- Obtém métricas dos experimentos MLflow
- Calcula médias de AUC, tempo de execução
- Conta runs bem-sucedidos vs falhados

**`check_api_health()`**
- Verifica saúde da API
- Testa endpoint /health
- Mede tempo de resposta

**`test_prediction_endpoint()`**
- Testa endpoint de predição
- Usa dados de teste
- Loga resultado no sistema de monitoramento

---

### 5. **model_pipeline.py** - Pipeline Avançado
**Pipeline mais complexo com feature engineering avançado**

#### Classe: `DecisionRecruitmentModel`

##### Métodos principais:

**`advanced_feature_engineering(self)`**
- Feature engineering mais sofisticado
- Cria features de compatibilidade técnica
- Análise de texto avançada
- Features temporais
- Features de engagement
- Scores de match global

**`optimize_hyperparameters(self, X, y, n_trials, val_size, use_mlflow)`**
- Otimização bayesiana com Optuna
- Integração com MLflow callback
- Holdout validation
- Salva resultados em JSON

**`train_model(self, X, y, test_size, val_size, use_mlflow, use_optimized_params)`**
- Treinamento com 3 folds (treino/validação/teste)
- Usa parâmetros otimizados se disponível
- Logging completo no MLflow
- Validação cruzada

**`identify_continuous_variables(self, X)`**
- Identifica variáveis contínuas vs discretas
- Critério: presença de valores decimais
- Usado para normalização seletiva

**`apply_normalization(self, X, continuous_vars, fit_scaler)`**
- Aplica StandardScaler apenas em variáveis contínuas
- Preserva variáveis discretas sem normalização
- Ajusta scaler apenas no treinamento

---

### 6. **mlflow_experiments.py** - Experimentos MLflow
**Script para executar experimentos comparativos**

#### Classe: `MLflowExperimentRunner`

##### Métodos principais:

**`run_hyperparameter_experiment(self, X, y)`**
- Executa grid search de hiperparâmetros
- Testa 20 combinações diferentes
- Loga cada experimento no MLflow
- Retorna melhores parâmetros

**`run_model_comparison_experiment(self, X, y)`**
- Compara diferentes algoritmos
- Testa: XGBoost, Random Forest, Logistic Regression
- Loga cada modelo no MLflow
- Retorna resultados comparativos

**`run_feature_importance_experiment(self, X, y)`**
- Analisa importância das features
- Loga top features como métricas
- Cria visualizações de importância

**`run_all_experiments(self)`**
- Executa suite completa de experimentos
- Orquestra todos os experimentos
- Gera relatório final

---

## 🔄 Fluxo de Execução Completo

### 1. **Preparação**
```bash
# Instalar dependências
pip install -r requirements.txt
```

### 2. **Execução do Pipeline**
```bash
# Executar pipeline completo
python main.py
```

**O que acontece:**
1. Carrega dados de `dataset_preparado.csv`
2. Aplica feature engineering
3. Executa otimização de hiperparâmetros (30 trials)
4. Treina modelo final com parâmetros otimizados
5. Salva modelo em `models/`
6. Registra modelo no MLflow
7. Loga todas as métricas e artefatos

### 3. **Inicialização da API**
```bash
# Iniciar API REST
python fastapi_app.py
```

**O que acontece:**
1. Carrega modelo treinado
2. Configura endpoints REST
3. Inicia servidor na porta 8000
4. Disponibiliza interface de predição

### 4. **Monitoramento**
```bash
# Executar monitoramento
python monitoring.py
```

**O que acontece:**
1. Verifica saúde da API
2. Testa endpoints
3. Calcula métricas de performance
4. Monitora MLflow
5. Gera relatório de status

### 5. **Visualização MLflow**
```bash
# Iniciar MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

**O que acontece:**
1. Inicia interface web do MLflow
2. Disponibiliza em http://localhost:5000
3. Mostra experimentos, runs, métricas
4. Permite navegar pelo Model Registry

---

## 📊 Resultados Esperados

### Métricas de Performance
- **AUC Score**: ~83% (após otimização)
- **CV AUC**: ~77% (±9%)
- **Features**: 30 features otimizadas
- **Tempo de treinamento**: ~2-3 minutos

### Arquivos Gerados
- `models/xgboost_model.pkl` - Modelo treinado
- `models/label_encoders.pkl` - Encoders categóricos
- `models/scaler.pkl` - Normalizador
- `models/model_metadata.json` - Metadados
- `optimization_results.json` - Parâmetros otimizados
- `mlruns/` - Tracking do MLflow

### API Endpoints
- `GET /health` - Status da aplicação
- `POST /predict` - Predição individual
- `POST /predict_batch` - Predição em lote
- `GET /model_info` - Informações do modelo
- `GET /feature_importance` - Importância das features
- `GET /mlflow_info` - Status do MLflow

---

## 🎯 Próximos Passos

1. **Deploy**: Usar `deploy.py` para deploy em produção
2. **Monitoramento**: Configurar alertas de drift
3. **Retreinamento**: Automatizar retreinamento periódico
4. **A/B Testing**: Implementar testes de modelos
5. **CI/CD**: Integrar com pipeline de CI/CD

---

## 🚨 Troubleshooting

### Problemas Comuns

**MLflow UI não inicia:**
```bash
# Tentar porta diferente
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

**API não carrega modelo:**
- Verificar se `models/` existe
- Verificar se `model_metadata.json` está presente
- Verificar logs da API

**Otimização falha:**
- Verificar se dataset existe
- Verificar memória disponível
- Reduzir `n_trials` se necessário

**Predições inconsistentes:**
- Verificar se feature engineering é idêntico
- Verificar se encoders estão sendo aplicados
- Verificar se normalização está sendo aplicada

---

## 📚 Referências

- [MLflow Documentation](https://mlflow.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
