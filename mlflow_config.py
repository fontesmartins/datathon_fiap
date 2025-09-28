#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuração MLflow para Decision Recruitment AI
Sistema de tracking, registry e experimentos
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MLflowConfig:
"""Configuração centralizada do MLflow"""

def __init__(self, 
experiment_name: str = "decision-recruitment-ai",
tracking_uri: str = "file:./mlruns",
registry_uri: str = None):
"""
Inicializa configuração MLflow

Args:
experiment_name: Nome do experimento
tracking_uri: URI do tracking server (local por padrão)
registry_uri: URI do model registry
"""
self.experiment_name = experiment_name
self.tracking_uri = tracking_uri
self.registry_uri = registry_uri

# Configurar MLflow
self.setup_mlflow()

def setup_mlflow(self):
"""Configura MLflow tracking e registry"""
try:
# Configurar tracking URI
mlflow.set_tracking_uri(self.tracking_uri)

# Configurar registry URI se fornecido
if self.registry_uri:
mlflow.set_registry_uri(self.registry_uri)

# Criar ou obter experimento
try:
experiment = mlflow.get_experiment_by_name(self.experiment_name)
if experiment is None:
experiment_id = mlflow.create_experiment(self.experiment_name)
logger.info(f"Experimento criado: {self.experiment_name} (ID: {experiment_id})")
else:
experiment_id = experiment.experiment_id
logger.info(f"Experimento encontrado: {self.experiment_name} (ID: {experiment_id})")
except Exception as e:
logger.warning(f"Erro ao configurar experimento: {e}")
experiment_id = "0" # Usar experimento padrão

# Definir experimento ativo
mlflow.set_experiment(self.experiment_name)

logger.info("MLflow configurado com sucesso!")

except Exception as e:
logger.error(f"Erro ao configurar MLflow: {e}")
raise

def start_run(self, run_name: str = None, tags: dict = None):
"""Inicia um novo run do MLflow"""
if run_name is None:
run_name = f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

if tags is None:
tags = {}

# Tags padrão
default_tags = {
"project": "decision-recruitment-ai",
"model_type": "xgboost",
"created_at": datetime.now().isoformat()
}
default_tags.update(tags)

return mlflow.start_run(run_name=run_name, tags=default_tags)

def log_model_params(self, params: dict):
"""Loga parâmetros do modelo"""
mlflow.log_params(params)
logger.info(f"Parâmetros logados: {len(params)} parâmetros")

def log_model_metrics(self, metrics: dict):
"""Loga métricas do modelo"""
mlflow.log_metrics(metrics)
logger.info(f"Métricas logadas: {list(metrics.keys())}")

def log_model_artifacts(self, artifacts: dict):
"""Loga artefatos (arquivos)"""
for name, path in artifacts.items():
if os.path.exists(path):
mlflow.log_artifact(path, name)
logger.info(f"Artefato logado: {name} -> {path}")
else:
logger.warning(f"Artefato não encontrado: {path}")

def log_xgboost_model(self, model, model_name: str = "xgboost_model", 
signature=None, input_example=None):
"""Loga modelo XGBoost"""
try:
mlflow.xgboost.log_model(
xgb_model=model,
artifact_path=model_name,
signature=signature,
input_example=input_example,
registered_model_name="decision-recruitment-model"
)
logger.info(f"Modelo XGBoost logado: {model_name}")
except Exception as e:
logger.error(f"Erro ao logar modelo XGBoost: {e}")
raise

def log_sklearn_model(self, model, model_name: str = "sklearn_model",
signature=None, input_example=None):
"""Loga modelo Scikit-learn"""
try:
mlflow.sklearn.log_model(
sk_model=model,
artifact_path=model_name,
signature=signature,
input_example=input_example,
registered_model_name="decision-recruitment-model"
)
logger.info(f"Modelo Scikit-learn logado: {model_name}")
except Exception as e:
logger.error(f"Erro ao logar modelo Scikit-learn: {e}")
raise

def log_feature_importance(self, feature_names: list, importance_scores: list):
"""Loga importância das features"""
try:
# Criar DataFrame com importância das features
import pandas as pd
feature_importance_df = pd.DataFrame({
'feature': feature_names,
'importance': importance_scores
}).sort_values('importance', ascending=False)

# Logar como artefato
importance_path = "feature_importance.csv"
feature_importance_df.to_csv(importance_path, index=False)
mlflow.log_artifact(importance_path, "feature_importance")

# Logar top features como métricas
top_features = feature_importance_df.head(10)
for idx, row in top_features.iterrows():
mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])

# Limpar arquivo temporário
os.remove(importance_path)

logger.info(f"Importância das features logada: {len(feature_names)} features")

except Exception as e:
logger.error(f"Erro ao logar importância das features: {e}")

def log_dataset_info(self, dataset_path: str, target_column: str = "contratado"):
"""Loga informações do dataset"""
try:
import pandas as pd

# Carregar dataset
df = pd.read_csv(dataset_path)

# Informações básicas
dataset_info = {
"dataset_rows": len(df),
"dataset_columns": len(df.columns),
"target_distribution": df[target_column].value_counts().to_dict() if target_column in df.columns else {},
"missing_values": df.isnull().sum().sum(),
"dataset_size_mb": os.path.getsize(dataset_path) / (1024 * 1024)
}

# Logar métricas
mlflow.log_metrics(dataset_info)

# Logar dataset como artefato
mlflow.log_artifact(dataset_path, "dataset")

logger.info(f"Informações do dataset logadas: {len(df)} linhas, {len(df.columns)} colunas")

except Exception as e:
logger.error(f"Erro ao logar informações do dataset: {e}")

def register_model(self, model_name: str = "decision-recruitment-model", 
model_version: str = None, stage: str = "Production"):
"""Registra modelo no Model Registry"""
try:
# Obter run atual
current_run = mlflow.active_run()
if current_run is None:
raise ValueError("Nenhum run ativo encontrado")

# Registrar modelo
model_uri = f"runs:/{current_run.info.run_id}/xgboost_model"

if model_version:
registered_model = mlflow.register_model(
model_uri=model_uri,
name=model_name,
tags={"version": model_version}
)
else:
registered_model = mlflow.register_model(
model_uri=model_uri,
name=model_name
)

# Transicionar para stage
if stage != "None":
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
name=model_name,
version=registered_model.version,
stage=stage
)

logger.info(f"Modelo registrado: {model_name} v{registered_model.version} -> {stage}")
return registered_model

except Exception as e:
logger.error(f"Erro ao registrar modelo: {e}")
raise

def load_model(self, model_name: str = "decision-recruitment-model", 
stage: str = "Production"):
"""Carrega modelo do Model Registry"""
try:
model_uri = f"models:/{model_name}/{stage}"
model = mlflow.xgboost.load_model(model_uri)
logger.info(f"Modelo carregado: {model_name} ({stage})")
return model

except Exception as e:
logger.error(f"Erro ao carregar modelo: {e}")
raise

# Instância global de configuração
mlflow_config = MLflowConfig()

def get_mlflow_config():
"""Retorna instância global de configuração MLflow"""
return mlflow_config
