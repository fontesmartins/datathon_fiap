# Decision Recruitment AI

Sistema de Inteligência Artificial para predição de match candidato-vaga usando XGBoost, desenvolvido para a Decision.

## Visão Geral

O **Decision Recruitment AI** é uma solução completa de Machine Learning que utiliza algoritmos de classificação para prever a probabilidade de contratação de candidatos, otimizando o processo de recrutamento da Decision.

### Objetivos

- **Automatizar** o processo de triagem inicial de candidatos
- **Otimizar** o match entre candidatos e vagas
- **Reduzir** o tempo de recrutamento
- **Aumentar** a precisão na seleção de candidatos
- **Fornecer** insights baseados em dados históricos

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        DADOS DE ENTRADA                        │
├─────────────────────────────────────────────────────────────────┤
│  vagas.json     prospects.json     applicants.json             │
│  14.081 vagas   14.222 prospecções  42.482 candidatos         │
└─────────────────┬───────────────────┬───────────────────────────┘
                  │                   │
                  └─────────┬─────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE TREINAMENTO                     │
├─────────────────────────────────────────────────────────────────┤
│  Dataset Unificado → Feature Engineering → Otimização          │
│  53.759 registros   28 features         Optuna + XGBoost      │
│                                            ↓                   │
│                                   Modelo Treinado              │
│                                   AUC: 0.7946                  │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────┐
│                      MODELO PERSISTIDO                         │
├─────────────────────────────────────────────────────────────────┤
│  xgboost_model.pkl    label_encoders.pkl                       │
│  scaler.pkl          model_metadata.json                       │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────┐
│                         API FASTAPI                            │
├─────────────────────────────────────────────────────────────────┤
│  /predict        /predict_batch     /health      /debug         │
│  Individual      Lote               Status       Troubleshoot   │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────┤
│  Railway (Produção)              Docker (Local/Staging)        │
│  https://decision-recruitment    Container local               │
│  .up.railway.app/                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MLFLOW TRACKING                           │
├─────────────────────────────────────────────────────────────────┤
│  Experimentos    Model Registry    Feature Importance          │
│  Métricas        Versionamento     Visualização                │
└─────────────────────────────────────────────────────────────────┘
```

## Dados

### Estrutura dos Dados

- **vagas.json**: 14.081 vagas com informações detalhadas
- **prospects.json**: 14.222 prospecções de candidatos
- **applicants.json**: 42.482 candidatos com perfis completos

### Dataset Unificado

- **53.759 registros** de candidatos-vagas
- **Taxa de contratação**: 5,55% (2.984 candidatos contratados de 50.775 total)
- **57 features** extraídas e preparadas
- **28 features finais** após feature engineering e otimização

## Features do Modelo

### Features de Compatibilidade
- Match de nível profissional
- Compatibilidade de idiomas (inglês/espanhol)
- Alinhamento de formação acadêmica

### Features de Análise de Texto
- Presença de palavras-chave técnicas no CV
- Certificações mencionadas
- Comprimento e qualidade do CV

### Features Temporais
- Tempo de resposta à vaga
- Categorização temporal

### Features de Engagement
- Qualidade dos comentários
- Presença de informações de remuneração
- Detalhamento das candidaturas

## Instalação e Uso

### 1. Clone o Repositório

```bash
git clone https://github.com/fontesmartins/datathon_fiap.git
cd fiap-final
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o Modelo

```bash
python main.py
```

### 4. Executar API Localmente

```bash
python fastapi_app.py
```

### 5. Deploy com Docker

```bash
./deploy_linux.sh
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints Disponíveis

#### 1. Health Check
```http
GET /health
```

#### 2. Predição Única
```http
POST /predict
Content-Type: application/json

{
"candidate": {
"nome": "João Silva",
"nivel_profissional_candidato": "Sênior",
"nivel_ingles_candidato": "Avançado",
"nivel_espanhol_candidato": "Intermediário",
"cv_text": "Desenvolvedor Python com 5 anos de experiência...",
"pcd": "Não",
"remuneracao": 8000.0,
"estado": "São Paulo"
},
"job": {
"titulo_vaga": "Desenvolvedor Python Sênior",
"nivel_profissional_vaga": "Sênior",
"nivel_ingles_vaga": "Avançado",
"nivel_espanhol_vaga": "Básico",
"vaga_sap": "Não",
"competencia_tecnicas": "Python, Django, Flask, AWS, Docker",
"tipo_contratacao": "CLT Full"
}
}
```

#### 3. Predição em Lote
```http
POST /predict_batch
Content-Type: application/json

{
"candidates": [...],
"job": {...}
}
```

#### 4. Informações do Modelo
```http
GET /model_info
```

#### 5. Importância das Features
```http
GET /feature_importance
```

### Resposta da API

```json
{
"prediction": 1,
"probability": 0.85,
"confidence": "High",
"recommendation": "RECOMENDADO",
"explanation": "Candidato tem alta compatibilidade com a vaga",
"timestamp": "2025-09-12T20:30:00",
"status": "success"
}
```

## Testes

### Testes da API
```bash
# Teste avançado da API
python test_api_advanced.py

# Teste simples via script
./test_api.sh
```


## Performance do Modelo

### Métricas de Treinamento (Último Treinamento)
- **AUC Score**: 0.7946 (79.46% de acurácia)
- **CV AUC Score**: 0.7512 (±0.0779) - Validação cruzada
- **Taxa de Contratação**: 5.55% (dataset desbalanceado)
- **Features Utilizadas**: 28 features otimizadas (removido data leakage)
- **Algoritmo**: XGBoost com otimização bayesiana (30 trials)
- **Data do Treinamento**: 28/09/2025
- **Melhorias**: Removidas variáveis `cliente` e `recrutador` para evitar data leakage

### Features Mais Importantes (Top 10)
1. `tipo_contratacao_encoded` - Tipo de contratação (12.25%)
2. `nivel_ingles_x_encoded` - Nível de inglês da vaga (10.15%)
3. `is_sp` - Candidato em São Paulo (7.21%)
4. `nivel_espanhol_x_encoded` - Nível de espanhol da vaga (5.37%)
5. `cv_has_technical_keywords` - CV com palavras-chave técnicas (4.36%)
6. `nivel_academico_x_encoded` - Nível acadêmico da vaga (4.30%)
7. `nivel_academico_y_encoded` - Nível acadêmico do candidato (4.08%)
8. `nivel_ingles_y_encoded` - Nível de inglês do candidato (4.07%)
9. `area_atuacao_encoded` - Área de atuação (4.03%)
10. `nivel_ingles_compatibility` - Compatibilidade de inglês (3.98%)

## Docker

### Construir Imagem
```bash
docker build -t decision-recruitment-ai:latest .
```

### Executar Container
```bash
docker run -d --name decision-recruitment-api -p 8000:8000 decision-recruitment-ai:latest
```

### Verificar Status
```bash
docker ps --filter name=decision-recruitment-api
```

### Ver Logs
```bash
docker logs decision-recruitment-api
```

## Status da API

### ✅ API Online e Funcionando
- **Railway**: https://decision-recruitment.up.railway.app/
- **Status**: ✅ Funcionando perfeitamente
- **Features**: ✅ 28 features corretas
- **Predições**: ✅ Funcionando (individual e lote)
- **Última atualização**: 29/09/2025

### Teste Rápido
```bash
# Testar API
curl -X POST https://decision-recruitment.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d @test_prediction.json

# Verificar saúde
curl https://decision-recruitment.up.railway.app/health
```

## Monitoramento

### Logs de Predições
- Localização: `logs/predictions.log`
- Formato: JSON estruturado
- Inclui: timestamp, predição, probabilidade, tempo de resposta

### Métricas
- Localização: `logs/metrics.json`
- Período: Últimas 24 horas
- Inclui: taxa de recomendação, confiança, tempo de resposta

### Detecção de Drift
- Comparação com baseline
- Alertas automáticos
- Threshold configurável

## Documentação da API

Acesse a documentação interativa em:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Estrutura do Projeto

```
fiap-final/
README.md # Este arquivo
requirements.txt # Dependências Python (inclui MLflow)
Dockerfile # Configuração Docker
.dockerignore # Arquivos ignorados no Docker
main.py # Pipeline de treinamento (com MLflow)
fastapi_app.py # API FastAPI (com MLflow logging)
test_api_advanced.py # Testes avançados da API
test_api.sh # Script de teste simples
# monitoring.py removido (não funcional)
deploy_linux.sh # Script de deploy para Linux
data_analysis.py # Análise inicial dos dados
mlflow_config.py # Configuração MLflow
# mlflow_experiments.py removido (obsoleto)
validate_docker_linux.py # Validador Docker para Linux
models/ # Modelos treinados
xgboost_model.pkl
label_encoders.pkl
scaler.pkl
model_metadata.json
mlruns/ # MLflow tracking data
0/ # Experimento padrão
[experiment_id]/ # Experimentos específicos
logs/ # Logs do sistema
predictions.log
metrics.json
dataset_preparado.csv # Dataset processado
```

## Casos de Uso

### 1. Triagem Automática
- Classificar candidatos automaticamente
- Priorizar candidatos com maior probabilidade
- Reduzir tempo de análise manual

### 2. Análise de Compatibilidade
- Avaliar match técnico
- Verificar alinhamento cultural
- Identificar gaps de competências

### 3. Insights de Recrutamento
- Identificar padrões de sucesso
- Otimizar descrições de vagas
- Melhorar processo de seleção

## MLflow Integration

O projeto agora inclui integração completa com MLflow para experimentação, tracking e versionamento de modelos.

### Iniciando MLflow

```bash
# Iniciar MLflow UI
python start_mlflow.py

# Ou manualmente
mlflow ui --backend-store-uri file:./mlruns
# Acesse: http://localhost:5000
```

### Funcionalidades MLflow

#### 1. **Experiment Tracking**
- Rastreamento automático de parâmetros e métricas
- Comparação de diferentes configurações de modelo
- Visualização de resultados em tempo real

#### 2. **Model Registry**
- Versionamento automático de modelos
- Gestão de stages (Staging, Production, Archived)
- Deploy automático do melhor modelo

#### 3. **Experimentos Automatizados**
```bash
# Executar suite completa de experimentos
# python mlflow_experiments.py (arquivo removido)

# Treinar modelo com MLflow tracking
python main.py
```

#### 4. **API Integration**
- Logging automático de predições em produção
- Métricas de performance em tempo real
- Endpoint `/mlflow_info` para informações do MLflow
- Endpoint `/debug` para troubleshooting

#### 5. **Monitoramento Avançado**
```bash
# Monitorar sistema completo (incluindo MLflow)
# python monitoring.py (arquivo removido)
```

### Experimentos Disponíveis

1. **Hyperparameter Tuning**: Testa diferentes configurações do XGBoost
2. **Model Comparison**: Compara XGBoost, Random Forest e Logistic Regression
3. **Feature Analysis**: Analisa importância das features
4. **Production Logging**: Registra predições em tempo real


## Próximos Passos

### Melhorias Planejadas
- [x] **MLflow Integration** - Experiment tracking e model registry ✅
- [ ] Interface web para usuários
- [ ] Modelo de recomendação de vagas
- [ ] Análise de sentimento em CVs
- [ ] Dashboard de métricas em tempo real

### Expansões
- [ ] Suporte a mais tipos de vaga
- [ ] Integração com LinkedIn API

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Execução Rápida

### 1. Treinamento do Modelo
```bash
python main.py
```
**Resultados esperados:**
- Modelo salvo em `models/`
- Otimização de hiperparâmetros (30 trials)
- Logs do MLflow em `mlruns/`
- AUC Score: ~0.79

### 2. Executar API
```bash
python fastapi_app.py
```
**Endpoints disponíveis:**
- `GET /health` - Health check
- `POST /predict` - Predição individual
- `POST /predict_batch` - Predição em lote
- `GET /model_info` - Informações do modelo
- `GET /feature_importance` - Importância das features
- `GET /mlflow_info` - Informações do MLflow
- `GET /debug` - Debug e troubleshooting

### 3. Testar API
```bash
# Health check
curl http://localhost:8000/health

# Predição
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_prediction.json
```

### 4. Deploy com Docker
```bash
./deploy_linux.sh
```

## Documentação Completa

Para informações detalhadas sobre arquitetura, uso da API, MLflow, troubleshooting e mais:

**[📚 Documentação Completa](DOCUMENTACAO.md)**

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Equipe

- **Desenvolvedor**: [Seu Nome]
- **Instituição**: FIAP

## Suporte

Para dúvidas ou suporte:
- Email: ezequiel.martins2@icloud.com
- LinkedIn: https://www.linkedin.com/in/fontesmartins23/
- GitHub: https://github.com/fontesmartins/

---