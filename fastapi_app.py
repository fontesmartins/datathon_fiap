#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API FastAPI para Modelo Preditivo de Recrutamento
Decision - Sistema de Match Candidato-Vaga
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import os
import mlflow
import mlflow.xgboost
from model_pipeline import DecisionRecruitmentModel
from mlflow_config import get_mlflow_config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
title="Decision Recruitment AI",
description="API para predição de match candidato-vaga usando XGBoost",
version="1.0.0",
docs_url="/docs",
redoc_url="/redoc"
)

# Carregar modelo globalmente
model = None
mlflow_config = None

# Modelos Pydantic para validação de dados
class CandidateData(BaseModel):
"""Modelo para dados do candidato"""
nome: str = Field(..., description="Nome do candidato")
nivel_profissional_candidato: str = Field(..., description="Nível profissional do candidato")
nivel_ingles_candidato: str = Field(..., description="Nível de inglês do candidato")
nivel_espanhol_candidato: str = Field(..., description="Nível de espanhol do candidato")
cv_text: str = Field(..., description="Texto do CV")
pcd: str = Field(default="Não", description="Pessoa com deficiência")
remuneracao: Optional[float] = Field(default=0, description="Remuneração esperada")
estado: str = Field(default="São Paulo", description="Estado de residência")

class JobData(BaseModel):
"""Modelo para dados da vaga"""
titulo_vaga: str = Field(..., description="Título da vaga")
nivel_profissional_vaga: str = Field(..., description="Nível profissional da vaga")
nivel_ingles_vaga: str = Field(..., description="Nível de inglês da vaga")
nivel_espanhol_vaga: str = Field(..., description="Nível de espanhol da vaga")
vaga_sap: str = Field(default="Não", description="É vaga SAP")
competencia_tecnicas: str = Field(..., description="Competências técnicas da vaga")
cliente: str = Field(..., description="Cliente solicitante")
tipo_contratacao: str = Field(..., description="Tipo de contratação")

class PredictionRequest(BaseModel):
"""Modelo para requisição de predição"""
candidate: CandidateData
job: JobData

class PredictionResponse(BaseModel):
"""Modelo para resposta de predição"""
prediction: int = Field(..., description="Predição (0 ou 1)")
probability: float = Field(..., description="Probabilidade de contratação")
confidence: str = Field(..., description="Nível de confiança")
recommendation: str = Field(..., description="Recomendação")
explanation: str = Field(..., description="Explicação da predição")
timestamp: str = Field(..., description="Timestamp da predição")
status: str = Field(..., description="Status da operação")

class BatchPredictionRequest(BaseModel):
"""Modelo para predição em lote"""
candidates: List[CandidateData]
job: JobData

class BatchPredictionResponse(BaseModel):
"""Modelo para resposta de predição em lote"""
results: List[Dict[str, Any]]
total_candidates: int
recommended_count: int
timestamp: str
status: str

class HealthResponse(BaseModel):
"""Modelo para resposta de health check"""
status: str
timestamp: str
model_loaded: bool

def load_model():
"""Carrega o modelo treinado"""
global model, mlflow_config
try:
# Inicializar MLflow config
mlflow_config = get_mlflow_config()

# Tentar carregar do MLflow Model Registry primeiro
try:
model = mlflow_config.load_model(
model_name="decision-recruitment-model",
stage="Production"
)
logger.info("Modelo carregado do MLflow Model Registry!")
return True
except Exception as e:
logger.warning(f"Erro ao carregar do MLflow Registry: {e}")
logger.info("Tentando carregar modelo local...")

# Fallback para modelo local
model = DecisionRecruitmentModel()
model.load_model('models/')
logger.info("Modelo carregado localmente com sucesso!")
return True

except Exception as e:
logger.error(f"Erro ao carregar modelo: {str(e)}")
return False

def preprocess_input(candidate_data: CandidateData, job_data: JobData) -> pd.DataFrame:
"""Preprocessa dados de entrada para predição"""
try:
# Criar DataFrame com dados combinados
data = {
'nivel_profissional_candidato': candidate_data.nivel_profissional_candidato,
'nivel_ingles_candidato': candidate_data.nivel_ingles_candidato,
'nivel_espanhol_candidato': candidate_data.nivel_espanhol_candidato,
'cv_text': candidate_data.cv_text,
'pcd': candidate_data.pcd,
'remuneracao': candidate_data.remuneracao,
'estado': candidate_data.estado,
'nivel_profissional_vaga': job_data.nivel_profissional_vaga,
'nivel_ingles_vaga': job_data.nivel_ingles_vaga,
'nivel_espanhol_vaga': job_data.nivel_espanhol_vaga,
'vaga_sap': job_data.vaga_sap,
'competencia_tecnicas': job_data.competencia_tecnicas,
'cliente': job_data.cliente,
'tipo_contratacao': job_data.tipo_contratacao
}

df = pd.DataFrame([data])

# Aplicar feature engineering
df = apply_basic_feature_engineering(df)

# Selecionar apenas features necessárias
feature_columns = model.feature_columns

# Criar DataFrame com todas as features necessárias, preenchendo com 0 as que não existem
df_features = pd.DataFrame(index=df.index)
for col in feature_columns:
if col in df.columns:
df_features[col] = df[col].fillna(0)
else:
df_features[col] = 0

return df_features

except Exception as e:
logger.error(f"Erro no preprocessamento: {str(e)}")
raise e

def apply_basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
"""Aplica feature engineering básico aos dados de entrada"""

# 1. Features de compatibilidade
nivel_mapping = {
'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Especialista': 4, 'Líder': 5
}

idioma_mapping = {
'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4
}

# Compatibilidade de nível profissional
if 'nivel_profissional_vaga' in df.columns and 'nivel_profissional_candidato' in df.columns:
df['nivel_profissional_vaga_score'] = df['nivel_profissional_vaga'].map(nivel_mapping).fillna(0)
df['nivel_profissional_candidato_score'] = df['nivel_profissional_candidato'].map(nivel_mapping).fillna(0)
df['nivel_profissional_compatibility'] = 1 - abs(df['nivel_profissional_vaga_score'] - df['nivel_profissional_candidato_score']) / 4

# Compatibilidade de inglês
if 'nivel_ingles_vaga' in df.columns and 'nivel_ingles_candidato' in df.columns:
df['nivel_ingles_vaga_score'] = df['nivel_ingles_vaga'].map(idioma_mapping).fillna(0)
df['nivel_ingles_candidato_score'] = df['nivel_ingles_candidato'].map(idioma_mapping).fillna(0)
df['nivel_ingles_compatibility'] = 1 - abs(df['nivel_ingles_vaga_score'] - df['nivel_ingles_candidato_score']) / 4

# Compatibilidade de espanhol
if 'nivel_espanhol_vaga' in df.columns and 'nivel_espanhol_candidato' in df.columns:
df['nivel_espanhol_vaga_score'] = df['nivel_espanhol_vaga'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_candidato_score'] = df['nivel_espanhol_candidato'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_compatibility'] = 1 - abs(df['nivel_espanhol_vaga_score'] - df['nivel_espanhol_candidato_score']) / 4

# 2. Features de texto
if 'cv_text' in df.columns:
df['cv_length'] = df['cv_text'].str.len().fillna(0)
df['cv_has_technical_keywords'] = df['cv_text'].str.contains(
'python|java|javascript|sql|aws|azure|docker|kubernetes|react|angular|node', 
case=False, na=False
).astype(int)
df['cv_has_certifications'] = df['cv_text'].str.contains(
'certificação|certificado|certified|aws|microsoft|google|oracle', 
case=False, na=False
).astype(int)

# 3. Features básicas
if 'vaga_sap' in df.columns:
df['is_sap_vaga'] = (df['vaga_sap'] == 'Sim').astype(int)

if 'pcd' in df.columns:
df['is_pcd'] = (df['pcd'] == 'Sim').astype(int)

if 'estado' in df.columns:
df['is_sp'] = df['estado'].str.contains('São Paulo', na=False).astype(int)

if 'remuneracao' in df.columns:
df['remuneracao_numeric'] = pd.to_numeric(df['remuneracao'], errors='coerce').fillna(0)

# 4. Features de match
if 'nivel_profissional_vaga' in df.columns and 'nivel_profissional_candidato' in df.columns:
df['nivel_profissional_match'] = (df['nivel_profissional_vaga'] == df['nivel_profissional_candidato']).astype(int)

if 'nivel_ingles_vaga' in df.columns and 'nivel_ingles_candidato' in df.columns:
df['nivel_ingles_match'] = (df['nivel_ingles_vaga'] == df['nivel_ingles_candidato']).astype(int)

if 'nivel_espanhol_vaga' in df.columns and 'nivel_espanhol_candidato' in df.columns:
df['nivel_espanhol_match'] = (df['nivel_espanhol_vaga'] == df['nivel_espanhol_candidato']).astype(int)

return df

@app.on_event("startup")
async def startup_event():
"""Evento de inicialização da aplicação"""
logger.info("Iniciando aplicação...")
if not load_model():
logger.error("Falha ao carregar modelo. Aplicação não iniciada.")
raise Exception("Modelo não pôde ser carregado")

@app.get("/", response_model=Dict[str, str])
async def root():
"""Endpoint raiz"""
return {
"message": "Decision Recruitment AI API",
"version": "1.0.0",
"docs": "/docs"
}

@app.get("/health", response_model=HealthResponse)
async def health_check():
"""Endpoint de health check"""
return HealthResponse(
status="healthy",
timestamp=datetime.now().isoformat(),
model_loaded=model is not None
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
"""Endpoint principal para predição"""
try:
# Verificar se modelo está carregado
if model is None:
raise HTTPException(status_code=500, detail="Modelo não carregado")

logger.info(f"Recebida requisição de predição para candidato: {request.candidate.nome}")

# Preprocessar dados
processed_data = preprocess_input(request.candidate, request.job)

# Fazer predição
if hasattr(model, 'model'):
# Modelo carregado localmente
prediction = model.model.predict(processed_data)[0]
probability = model.model.predict_proba(processed_data)[0][1]
else:
# Modelo carregado do MLflow
prediction = model.predict(processed_data)[0]
probability = model.predict_proba(processed_data)[0][1]

# Determinar confiança
if probability > 0.8:
confidence = 'High'
elif probability > 0.5:
confidence = 'Medium'
else:
confidence = 'Low'

# Determinar recomendação
if prediction == 1:
recommendation = 'RECOMENDADO'
explanation = 'Candidato tem alta compatibilidade com a vaga'
else:
recommendation = 'NÃO RECOMENDADO'
explanation = 'Candidato não atende aos critérios ideais para a vaga'

response = PredictionResponse(
prediction=int(prediction),
probability=float(probability),
confidence=confidence,
recommendation=recommendation,
explanation=explanation,
timestamp=datetime.now().isoformat(),
status='success'
)

# Logar predição no MLflow se disponível
if mlflow_config:
try:
with mlflow.start_run(run_name=f"prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
mlflow.log_params({
"candidate_name": request.candidate.nome,
"job_title": request.job.titulo_vaga,
"nivel_profissional_candidato": request.candidate.nivel_profissional_candidato,
"nivel_profissional_vaga": request.job.nivel_profissional_vaga,
"cliente": request.job.cliente
})

mlflow.log_metrics({
"prediction": int(prediction),
"probability": float(probability),
"confidence_score": 1.0 if confidence == 'High' else 0.5 if confidence == 'Medium' else 0.0
})

mlflow.log_text(
f"Recomendação: {recommendation}\nExplicação: {explanation}",
"prediction_details.txt"
)

except Exception as e:
logger.warning(f"Erro ao logar predição no MLflow: {e}")

logger.info(f"Predição realizada: {response.recommendation} (prob: {response.probability:.3f})")

return response

except Exception as e:
logger.error(f"Erro na predição: {str(e)}")
raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
"""Endpoint para predição em lote"""
try:
if model is None:
raise HTTPException(status_code=500, detail="Modelo não carregado")

candidates = request.candidates
job = request.job

if not candidates:
raise HTTPException(status_code=400, detail="Lista de candidatos vazia")

logger.info(f"Recebida requisição de predição em lote: {len(candidates)} candidatos")

# Processar cada candidato
results = []
for i, candidate in enumerate(candidates):
try:
# Preprocessar dados
processed_data = preprocess_input(candidate, job)

# Fazer predição
prediction = model.model.predict(processed_data)[0]
probability = model.model.predict_proba(processed_data)[0][1]

# Determinar confiança e recomendação
confidence = 'High' if probability > 0.8 else 'Medium' if probability > 0.5 else 'Low'
recommendation = 'RECOMENDADO' if prediction == 1 else 'NÃO RECOMENDADO'

results.append({
'candidate_index': i,
'candidate_name': candidate.nome,
'prediction': int(prediction),
'probability': float(probability),
'confidence': confidence,
'recommendation': recommendation
})

except Exception as e:
logger.error(f"Erro ao processar candidato {i}: {str(e)}")
results.append({
'candidate_index': i,
'candidate_name': candidate.nome,
'error': str(e)
})

response = BatchPredictionResponse(
results=results,
total_candidates=len(candidates),
recommended_count=sum(1 for r in results if r.get('prediction') == 1),
timestamp=datetime.now().isoformat(),
status='success'
)

logger.info(f"Predição em lote realizada: {len(candidates)} candidatos, {response.recommended_count} recomendados")

return response

except Exception as e:
logger.error(f"Erro na predição em lote: {str(e)}")
raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
"""Endpoint para informações do modelo"""
try:
if model is None:
raise HTTPException(status_code=500, detail="Modelo não carregado")

# Carregar metadados
with open('models/model_metadata.json', 'r') as f:
metadata = json.load(f)

return {
'model_type': metadata.get('model_type', 'XGBoost'),
'training_date': metadata.get('training_date'),
'feature_count': len(model.feature_columns),
'features': model.feature_columns[:10], # Primeiras 10 features
'categorical_features': len(model.categorical_columns),
'is_trained': metadata.get('is_trained', False),
'timestamp': datetime.now().isoformat(),
'status': 'success'
}

except Exception as e:
logger.error(f"Erro ao obter informações do modelo: {str(e)}")
raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature_importance")
async def feature_importance():
"""Endpoint para importância das features"""
try:
if model is None:
raise HTTPException(status_code=500, detail="Modelo não carregado")

# Obter importância das features
if hasattr(model, 'model'):
importance = model.model.feature_importances_
feature_names = model.feature_columns
else:
# Modelo do MLflow
importance = model.feature_importances_
feature_names = model.feature_names_in_

# Criar lista de features com importância
feature_importance_list = []
for name, imp in zip(feature_names, importance):
feature_importance_list.append({
'feature': name,
'importance': float(imp)
})

# Ordenar por importância
feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)

return {
'feature_importance': feature_importance_list[:20], # Top 20
'timestamp': datetime.now().isoformat(),
'status': 'success'
}

except Exception as e:
logger.error(f"Erro ao obter importância das features: {str(e)}")
raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow_info")
async def mlflow_info():
"""Endpoint para informações do MLflow"""
try:
if mlflow_config is None:
raise HTTPException(status_code=500, detail="MLflow não configurado")

# Obter informações do experimento
experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)

if experiment is None:
return {
'experiment_name': mlflow_config.experiment_name,
'experiment_id': None,
'tracking_uri': mlflow_config.tracking_uri,
'status': 'experiment_not_found'
}

# Obter runs recentes
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
experiment_ids=[experiment.experiment_id],
max_results=5,
order_by=["start_time DESC"]
)

recent_runs = []
for run in runs:
recent_runs.append({
'run_id': run.info.run_id,
'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
'start_time': run.info.start_time,
'status': run.info.status,
'metrics': {k: v for k, v in run.data.metrics.items() if k in ['auc_score', 'cv_auc_mean']}
})

return {
'experiment_name': experiment.name,
'experiment_id': experiment.experiment_id,
'tracking_uri': mlflow_config.tracking_uri,
'recent_runs': recent_runs,
'total_runs': len(client.search_runs(experiment_ids=[experiment.experiment_id])),
'timestamp': datetime.now().isoformat(),
'status': 'success'
}

except Exception as e:
logger.error(f"Erro ao obter informações do MLflow: {str(e)}")
raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
