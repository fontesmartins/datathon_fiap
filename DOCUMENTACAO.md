# Documentação Completa - Sistema de Recrutamento Inteligente

## Visão Geral

Este é um sistema de Machine Learning para predição de compatibilidade entre candidatos e vagas de emprego, desenvolvido para otimizar o processo de recrutamento. O sistema utiliza XGBoost com otimização bayesiana para fazer predições precisas sobre a adequação de candidatos a posições específicas.

## Arquitetura do Sistema

### Componentes Principais

1. **Pipeline de Treinamento** (`main.py`)
   - Carregamento e preparação dos dados
   - Feature engineering e codificação
   - Treinamento do modelo XGBoost
   - Otimização de hiperparâmetros com Optuna
   - Persistência do modelo e componentes

2. **API REST** (`fastapi_app.py`)
   - Endpoints para predição individual e em lote
   - Validação de dados com Pydantic
   - Logging de predições no MLflow
   - Health checks e informações do modelo

3. **Configuração MLflow** (`mlflow_config.py`)
   - Centralização das configurações do MLflow
   - Logging de experimentos e métricas
   - Persistência de feature importance

4. **Análise de Dados** (`data_analysis.py`)
   - Análise exploratória dos dados originais
   - Geração de insights e estatísticas

## Fluxo de Execução

### 1. Preparação do Ambiente

```bash
# Instalar dependências
pip install -r requirements.txt

# Verificar estrutura do projeto
python validate_docker_linux.py
```

### 2. Treinamento do Modelo

```bash
# Executar pipeline completo
python main.py
```

**Processo de Treinamento:**
- Carregamento do dataset `dataset_preparado.csv`
- Separação em features categóricas e numéricas
- Codificação de features categóricas (LabelEncoder)
- Normalização de features numéricas (StandardScaler)
- Criação de features de compatibilidade
- Treinamento com validação cruzada (5-fold)
- Otimização de hiperparâmetros (30 trials)
- Persistência do modelo e componentes

**Resultados Esperados:**
- Modelo salvo em `models/xgboost_model.pkl`
- Encoders salvos em `models/label_encoders.pkl`
- Scaler salvo em `models/scaler.pkl`
- Métricas e logs no MLflow (`mlruns/`)
- Feature importance em `feature_importance/`

### 3. Execução da API

```bash
# Iniciar servidor local
python fastapi_app.py
```

**Endpoints Disponíveis:**
- `GET /health` - Status da aplicação
- `POST /predict` - Predição individual
- `POST /predict_batch` - Predição em lote
- `GET /model_info` - Informações do modelo
- `GET /feature_importance` - Importância das features
- `GET /mlflow_info` - Informações do MLflow

### 4. Testes

```bash
# Teste avançado da API
python test_api_advanced.py

# Teste simples via script
./test_api.sh
```

### 5. Deploy com Docker

```bash
# Deploy automatizado
./deploy_linux.sh
```

**Processo de Deploy:**
- Build da imagem Docker
- Criação do container
- Verificação de saúde
- Exposição da porta 8000

## Performance do Modelo

### Métricas Atuais (Último Treinamento)

- **AUC Score**: 0.7946 (79.46% de acurácia)
- **CV AUC Score**: 0.7512 (±0.0779) - Validação cruzada
- **Taxa de Contratação**: 5.55% (dataset desbalanceado)
- **Features Utilizadas**: 28 features otimizadas
- **Algoritmo**: XGBoost com otimização bayesiana
- **Data do Treinamento**: 28/09/2025

### Features Mais Importantes (Top 10)

1. `tipo_contratacao_encoded` - Tipo de contratação (11.65%)
2. `nivel_ingles_x_encoded` - Nível de inglês da vaga (9.17%)
3. `nivel_espanhol_match` - Compatibilidade de espanhol (7.18%)
4. `nivel_espanhol_x_encoded` - Nível de espanhol da vaga (6.31%)
5. `cidade_encoded` - Cidade codificada (5.78%)
6. `nivel_academico_x_encoded` - Nível acadêmico da vaga (5.10%)
7. `is_sp` - Candidato em São Paulo (4.79%)
8. `nivel_academico_y_encoded` - Nível acadêmico do candidato (4.51%)
9. `nivel_ingles_y_encoded` - Nível de inglês do candidato (3.52%)
10. `cv_has_technical_keywords` - CV com palavras-chave técnicas (3.49%)

## Estrutura de Dados

### Dataset de Entrada

O sistema utiliza o arquivo `dataset_preparado.csv` com as seguintes colunas principais:

**Features da Vaga:**
- `nivel_profissional_x` - Nível profissional da vaga
- `nivel_academico_x` - Nível acadêmico da vaga
- `nivel_ingles_x` - Nível de inglês da vaga
- `nivel_espanhol_x` - Nível de espanhol da vaga
- `area_atuacao` - Área de atuação
- `cidade` - Cidade da vaga
- `tipo_contratacao` - Tipo de contratação
- `titulo_profissional` - Título da vaga

**Features do Candidato:**
- `nivel_profissional_y` - Nível profissional do candidato
- `nivel_academico_y` - Nível acadêmico do candidato
- `nivel_ingles_y` - Nível de inglês do candidato
- `nivel_espanhol_y` - Nível de espanhol do candidato

**Features Computadas:**
- `nivel_profissional_compatibility` - Compatibilidade de nível profissional
- `nivel_academico_compatibility` - Compatibilidade de nível acadêmico
- `nivel_ingles_compatibility` - Compatibilidade de inglês
- `nivel_espanhol_compatibility` - Compatibilidade de espanhol
- `is_sp` - Candidato em São Paulo
- `cv_has_technical_keywords` - CV com palavras-chave técnicas

**Target:**
- `contratado` - Variável alvo (0 ou 1)

### Prevenção de Data Leakage

O modelo foi otimizado para evitar data leakage **NÃO UTILIZANDO** as seguintes colunas do dataset:
- `cliente` - Informação do cliente da vaga (presente no dataset, mas não usada)
- `recrutador` - Informação do recrutador (presente no dataset, mas não usada)

Essas variáveis existem no dataset original, mas foram **EXCLUÍDAS** do treinamento porque não estariam disponíveis em cenários reais de predição.

## API - Uso Detalhado

### Predição Individual

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "candidate": {
    "nome": "João Silva",
    "nivel_profissional_candidato": "Pleno",
    "nivel_ingles_candidato": "Intermediário",
    "nivel_espanhol_candidato": "Básico",
    "estado": "São Paulo",
    "cv_text": "Experiência em Python, Machine Learning, SQL",
    "remuneracao": 8000,
    "pcd": "Não"
  },
  "job": {
    "titulo_vaga": "Desenvolvedor Python",
    "nivel_profissional_vaga": "Pleno",
    "nivel_ingles_vaga": "Intermediário",
    "nivel_espanhol_vaga": "Básico",
    "vaga_sap": "Não",
    "competencia_tecnicas": "Python, Machine Learning",
    "tipo_contratacao": "CLT"
  }
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "confidence_score": 0.85,
  "candidate_name": "João Silva",
  "job_title": "Desenvolvedor Python"
}
```

### Predição em Lote

**Endpoint:** `POST /predict_batch`

**Request Body:**
```json
[
  {
    "candidate": { ... },
    "job": { ... }
  },
  {
    "candidate": { ... },
    "job": { ... }
  }
]
```

**Response:**
```json
[
  {
    "prediction": 1,
    "probability": 0.85,
    "confidence_score": 0.85,
    "candidate_name": "João Silva",
    "job_title": "Desenvolvedor Python"
  }
]
```

## MLflow Integration

### Experimentos

O sistema utiliza MLflow para:
- Tracking de experimentos
- Logging de métricas e parâmetros
- Persistência de modelos
- Visualização de resultados

### Estrutura de Logs

```
mlruns/
├── 0/
│   └── meta.yaml
└── [experiment_id]/
    ├── [run_id]/
    │   ├── artifacts/
    │   │   ├── feature_importance/
    │   │   └── xgboost_model/
    │   ├── metrics/
    │   ├── params/
    │   └── tags/
    └── meta.yaml
```

### Métricas Registradas

- `auc_score` - AUC do modelo
- `cv_auc_mean` - Média do CV AUC
- `cv_auc_std` - Desvio padrão do CV AUC
- `cv_auc_max` - Máximo CV AUC
- `cv_auc_min` - Mínimo CV AUC
- Feature importance de cada feature

## Monitoramento e Validação

### Validação Docker

```bash
python validate_docker_linux.py
```

Verifica:
- Configuração do Dockerfile
- Dependências no requirements.txt
- Arquivos ignorados no .dockerignore
- Presença de arquivos necessários

### Health Check

```bash
curl http://localhost:8000/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-09-28T10:30:00"
}
```

## Troubleshooting

### Problemas Comuns

1. **Erro de importação**
   - Verificar se todas as dependências estão instaladas
   - Executar `pip install -r requirements.txt`

2. **Modelo não encontrado**
   - Executar `python main.py` para treinar o modelo
   - Verificar se os arquivos estão em `models/`

3. **Erro de porta**
   - Verificar se a porta 8000 está livre
   - Usar `lsof -i :8000` para verificar

4. **Docker não funciona**
   - Executar `python validate_docker_linux.py`
   - Verificar se Docker está instalado

### Logs

- **API**: Logs no console durante execução
- **MLflow**: Logs em `mlruns/`
- **Modelo**: Logs durante treinamento em `main.py`

## Estrutura de Arquivos

```
fiap-final/
├── README.md                    # Documentação principal
├── DOCUMENTACAO.md              # Esta documentação
├── requirements.txt             # Dependências Python
├── Dockerfile                   # Configuração Docker
├── .dockerignore               # Arquivos ignorados no Docker
├── main.py                     # Pipeline de treinamento
├── fastapi_app.py              # API FastAPI
├── mlflow_config.py            # Configuração MLflow
├── data_analysis.py            # Análise inicial dos dados
├── test_api_advanced.py        # Testes avançados da API
├── test_api.sh                 # Script de teste simples
├── validate_docker_linux.py    # Validador Docker para Linux
├── deploy_linux.sh             # Script de deploy para Linux
├── dataset_preparado.csv       # Dataset principal
├── models/                     # Modelos treinados
│   ├── xgboost_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── feature_importance/         # Feature importance
│   └── feature_importance.csv
├── mlruns/                     # Logs do MLflow
└── test_*.json                 # Arquivos de teste
```

## Considerações Técnicas

### Otimizações Implementadas

1. **Feature Engineering**
   - Criação de features de compatibilidade
   - Detecção de palavras-chave técnicas
   - Normalização de features numéricas

2. **Prevenção de Overfitting**
   - Validação cruzada (5-fold)
   - Otimização de hiperparâmetros
   - Remoção de data leakage

3. **Performance**
   - Uso de XGBoost (gradient boosting)
   - Otimização bayesiana com Optuna
   - Persistência eficiente com joblib

### Limitações Conhecidas

1. **Dataset Desbalanceado**
   - Taxa de contratação baixa (5.55%)
   - Pode afetar métricas de precisão

2. **Features Limitadas**
   - Baseado apenas em dados estruturados
   - Não considera análise de sentimento de CV

3. **Temporalidade**
   - Modelo treinado em dados específicos
   - Pode precisar de retreinamento periódico

## Próximos Passos

### Melhorias Sugeridas

1. **Dados**
   - Coleta de mais dados históricos
   - Inclusão de feedback pós-contratação

2. **Modelo**
   - Experimentação com outros algoritmos
   - Ensemble de modelos
   - Feature selection automática

3. **Sistema**
   - Implementação de cache
   - Rate limiting na API
   - Monitoramento em tempo real

4. **Deploy**
   - CI/CD pipeline
   - Container orchestration (Kubernetes)
   - Load balancing

---

**Versão:** 1.0  
**Última Atualização:** 28/09/2025  
**Autor:** Sistema de Recrutamento Inteligente
