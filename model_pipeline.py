#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Completa de Treinamento do Modelo XGBoost
Decision - Sistema de Match Candidato-Vaga
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import warnings
import logging
from datetime import datetime
import re
import json
import os
import mlflow
import mlflow.xgboost
import optuna
from optuna.integration.mlflow import MLflowCallback
from mlflow_config import get_mlflow_config

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecisionRecruitmentModel:
"""
Modelo XGBoost para predição de match candidato-vaga
Foco: Identificar candidatos com maior probabilidade de contratação
"""

def __init__(self):
self.model = None
self.label_encoders = {}
self.scaler = StandardScaler()
self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='portuguese')
self.feature_columns = []
self.categorical_columns = []
self.text_columns = []
self.is_trained = False

def load_data(self, file_path):
"""Carrega o dataset preparado"""
logger.info(f"Carregando dados de: {file_path}")
self.df = pd.read_csv(file_path)
logger.info(f"Dados carregados: {len(self.df)} registros, {len(self.df.columns)} colunas")
return self.df

def advanced_feature_engineering(self):
"""Engenharia de features avançada para match candidato-vaga"""
logger.info("Iniciando feature engineering avançado...")

df = self.df.copy()

# 1. FEATURES DE COMPATIBILIDADE TÉCNICA
logger.info("Criando features de compatibilidade técnica...")

# Match de níveis (com scoring)
nivel_mapping = {
'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Especialista': 4, 'Líder': 5
}

df['nivel_profissional_vaga_score'] = df['nivel_profissional_x'].map(nivel_mapping).fillna(0)
df['nivel_profissional_candidato_score'] = df['nivel_profissional_y'].map(nivel_mapping).fillna(0)
df['nivel_profissional_compatibility'] = 1 - abs(df['nivel_profissional_vaga_score'] - df['nivel_profissional_candidato_score']) / 4

# Match de idiomas (com scoring)
idioma_mapping = {
'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4
}

df['nivel_ingles_vaga_score'] = df['nivel_ingles_x'].map(idioma_mapping).fillna(0)
df['nivel_ingles_candidato_score'] = df['nivel_ingles_y'].map(idioma_mapping).fillna(0)
df['nivel_ingles_compatibility'] = 1 - abs(df['nivel_ingles_vaga_score'] - df['nivel_ingles_candidato_score']) / 4

df['nivel_espanhol_vaga_score'] = df['nivel_espanhol_x'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_candidato_score'] = df['nivel_espanhol_y'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_compatibility'] = 1 - abs(df['nivel_espanhol_vaga_score'] - df['nivel_espanhol_candidato_score']) / 4

# 2. FEATURES DE ANÁLISE DE TEXTO
logger.info("Criando features de análise de texto...")

# Análise do CV
df['cv_has_technical_keywords'] = df['cv_pt'].str.contains(
'python|java|javascript|sql|aws|azure|docker|kubernetes|react|angular|node', 
case=False, na=False
).astype(int)

df['cv_has_management_keywords'] = df['cv_pt'].str.contains(
'gerenciamento|gestão|liderança|coordenador|supervisor|manager', 
case=False, na=False
).astype(int)

df['cv_has_certifications'] = df['cv_pt'].str.contains(
'certificação|certificado|certified|aws|microsoft|google|oracle', 
case=False, na=False
).astype(int)

# Análise das competências técnicas da vaga
df['vaga_has_cloud'] = df['competencia_tecnicas'].str.contains(
'aws|azure|gcp|cloud|nuvem', case=False, na=False
).astype(int)

df['vaga_has_ai_ml'] = df['competencia_tecnicas'].str.contains(
'machine learning|inteligência artificial|ai|ml|deep learning|tensorflow|pytorch', 
case=False, na=False
).astype(int)

df['vaga_has_devops'] = df['competencia_tecnicas'].str.contains(
'devops|ci/cd|jenkins|docker|kubernetes|terraform', 
case=False, na=False
).astype(int)

# 3. FEATURES TEMPORAIS AVANÇADAS
logger.info("Criando features temporais...")

# Converter datas
df['data_candidatura'] = pd.to_datetime(df['data_candidatura'], errors='coerce')
df['data_requisicao'] = pd.to_datetime(df['data_requisicao'], errors='coerce')

# Tempo de resposta
df['tempo_resposta_dias'] = (df['data_candidatura'] - df['data_requisicao']).dt.days
df['tempo_resposta_categoria'] = pd.cut(
df['tempo_resposta_dias'], 
bins=[-np.inf, 0, 1, 7, 30, np.inf], 
labels=['Antecipado', 'Mesmo dia', '1-7 dias', '1-4 semanas', 'Mais de 1 mês']
)

# 4. FEATURES DE ENGAGEMENT
logger.info("Criando features de engagement...")

# Comprimento e qualidade do comentário
df['comentario_length'] = df['comentario'].str.len().fillna(0)
df['has_detailed_comment'] = (df['comentario_length'] > 50).astype(int)

# Presença de informações de remuneração no comentário
df['comentario_has_salary'] = df['comentario'].str.contains(
r'r\$|salário|remuneração|valor', case=False, na=False
).astype(int)

# 5. FEATURES DE PERFIL DO CANDIDATO
logger.info("Criando features de perfil...")

# Experiência baseada no CV
df['cv_mentions_experience'] = df['cv_pt'].str.contains(
'anos|experiência|experiência|trabalho|empresa', case=False, na=False
).astype(int)

# Presença de formação
df['has_education'] = df['nivel_academico_y'].notna().astype(int)

# 6. FEATURES DE CONTEXTO DA VAGA
logger.info("Criando features de contexto da vaga...")

# Tipo de contratação
df['is_clt'] = df['tipo_contratacao'].str.contains('CLT', case=False, na=False).astype(int)
df['is_pj'] = df['tipo_contratacao'].str.contains('PJ', case=False, na=False).astype(int)

# 7. FEATURES DE MATCH GLOBAL
logger.info("Criando score de match global...")

# Score de compatibilidade geral
compatibility_features = [
'nivel_profissional_compatibility',
'nivel_ingles_compatibility', 
'nivel_espanhol_compatibility'
]

df['global_compatibility_score'] = df[compatibility_features].mean(axis=1)

# Score de fit técnico
technical_features = [
'cv_has_technical_keywords',
'cv_has_certifications',
'cv_mentions_experience'
]

df['technical_fit_score'] = df[technical_features].mean(axis=1)

# Score de engagement
engagement_features = [
'has_detailed_comment',
'comentario_has_salary',
'cv_length'
]

df['engagement_score'] = df[engagement_features].mean(axis=1)

self.df = df
logger.info("Feature engineering concluído!")

return df

def prepare_features(self):
"""Prepara features para treinamento"""
logger.info("Preparando features para treinamento...")

# Definir colunas categóricas
self.categorical_columns = [
'cliente', 'nivel_profissional_x', 'nivel_academico_x', 'nivel_ingles_x', 
'nivel_espanhol_x', 'area_atuacao', 'cidade', 'tipo_contratacao',
'titulo_profissional', 'nivel_profissional_y', 'nivel_academico_y',
'nivel_ingles_y', 'nivel_espanhol_y', 'recrutador', 'tempo_resposta_categoria'
]

# Definir colunas numéricas
self.feature_columns = [
'is_sap_vaga', 'nivel_profissional_match', 'nivel_academico_match',
'nivel_ingles_match', 'nivel_espanhol_match', 'cv_length', 'has_cv_en',
'dias_entre_requisicao_candidatura', 'is_pcd', 'is_sp', 'remuneracao_numeric',
'nivel_profissional_compatibility', 'nivel_ingles_compatibility', 
'nivel_espanhol_compatibility', 'cv_has_technical_keywords',
'cv_has_management_keywords', 'cv_has_certifications', 'vaga_has_cloud',
'vaga_has_ai_ml', 'vaga_has_devops', 'tempo_resposta_dias',
'comentario_length', 'has_detailed_comment', 'comentario_has_salary',
'cv_mentions_experience', 'has_education', 'is_clt', 'is_pj',
'global_compatibility_score', 'technical_fit_score', 'engagement_score'
]

# Filtrar colunas que existem no dataset
self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
self.categorical_columns = [col for col in self.categorical_columns if col in self.df.columns]

logger.info(f"Features numéricas: {len(self.feature_columns)}")
logger.info(f"Features categóricas: {len(self.categorical_columns)}")

return self.feature_columns, self.categorical_columns

def encode_categorical_features(self):
"""Codifica features categóricas"""
logger.info("Codificando features categóricas...")

for col in self.categorical_columns:
if col in self.df.columns:
le = LabelEncoder()
# Tratar valores nulos - converter para string primeiro
self.df[col] = self.df[col].astype(str).fillna('Unknown')
self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
self.label_encoders[col] = le
self.feature_columns.append(f'{col}_encoded')

logger.info("Features categóricas codificadas!")

def identify_continuous_variables(self, X):
"""
Identifica variáveis contínuas baseado no critério:
Uma variável é contínua se qualquer divisão por 1 resulta em resto diferente de zero
"""
logger.info("Identificando variáveis contínuas...")

continuous_vars = []
discrete_vars = []

for col in X.columns:
# Verificar se a coluna tem valores decimais
has_decimals = False

# Testar com uma amostra dos dados
sample_values = X[col].dropna().head(1000) # Amostra para performance

for value in sample_values:
if pd.notna(value) and isinstance(value, (int, float)):
# Verificar se tem parte decimal
if value != int(value):
has_decimals = True
break

if has_decimals:
continuous_vars.append(col)
logger.info(f" {col}: CONTÍNUA (tem valores decimais)")
else:
discrete_vars.append(col)
logger.info(f" {col}: DISCRETA (apenas valores inteiros)")

logger.info(f"Variáveis contínuas identificadas: {len(continuous_vars)}")
logger.info(f"Variáveis discretas identificadas: {len(discrete_vars)}")

return continuous_vars, discrete_vars

def apply_normalization(self, X, continuous_vars, fit_scaler=True):
"""
Aplica normalização (StandardScaler) apenas nas variáveis contínuas

Args:
X: DataFrame com features
continuous_vars: Lista de variáveis contínuas
fit_scaler: Se True, ajusta o scaler; se False, apenas transforma

Returns:
DataFrame com variáveis contínuas normalizadas
"""
if not continuous_vars:
logger.info("Nenhuma variável contínua encontrada. Pulando normalização.")
return X

logger.info(f"Aplicando normalização em {len(continuous_vars)} variáveis contínuas...")

X_normalized = X.copy()

if fit_scaler:
# Ajustar o scaler apenas nas variáveis contínuas
self.scaler.fit(X[continuous_vars])
logger.info("Scaler ajustado com dados de treinamento")

# Aplicar transformação apenas nas variáveis contínuas
X_normalized[continuous_vars] = self.scaler.transform(X[continuous_vars])

logger.info("Normalização aplicada com sucesso!")

return X_normalized

def prepare_training_data(self):
"""Prepara dados para treinamento"""
logger.info("Preparando dados para treinamento...")

# Filtrar apenas features numéricas válidas
valid_features = [col for col in self.feature_columns if col in self.df.columns]

# Remover linhas com target nulo
df_clean = self.df.dropna(subset=['contratado'])

# Separar features e target
X = df_clean[valid_features].fillna(0)
y = df_clean['contratado']

# Identificar variáveis contínuas
continuous_vars, discrete_vars = self.identify_continuous_variables(X)

# Aplicar normalização nas variáveis contínuas
X = self.apply_normalization(X, continuous_vars, fit_scaler=True)

logger.info(f"Dataset final: {X.shape[0]} amostras, {X.shape[1]} features")
logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")

return X, y

def optimize_hyperparameters(self, X, y, n_trials=100, val_size=0.2, use_mlflow=True):
"""
Otimização bayesiana de hiperparâmetros usando Optuna com Holdout Validation

Args:
X: Features de treinamento
y: Target
n_trials: Número de trials para otimização
val_size: Proporção do dataset para validação (holdout)
use_mlflow: Se deve usar MLflow para tracking

Returns:
dict: Melhores parâmetros encontrados
"""
logger.info(f"Iniciando otimização bayesiana com {n_trials} trials usando Holdout Validation...")

# Dividir dados uma vez para todos os trials (treino + validação)
X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=val_size, random_state=42, stratify=y
)

logger.info(f"Holdout split: {len(X_train)} treino, {len(X_val)} validação")

def objective(trial):
"""Função objetivo para o Optuna"""

# Definir espaço de busca dos hiperparâmetros
params = {
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth': trial.suggest_int('max_depth', 3, 10),
'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
'subsample': trial.suggest_float('subsample', 0.6, 1.0),
'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
'gamma': trial.suggest_float('gamma', 0.0, 5.0),
'random_state': 42,
'n_jobs': -1
}

# Treinar modelo com holdout validation
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Avaliar no conjunto de validação
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_pred_proba)

# Retornar o score (Optuna maximiza)
return auc_score

# Configurar estudo Optuna
study = optuna.create_study(
direction='maximize',
study_name='xgboost_hyperparameter_optimization',
load_if_exists=True
)

# Configurar callback MLflow se solicitado
if use_mlflow:
mlflow_callback = MLflowCallback(
tracking_uri="file:./mlruns",
metric_name="holdout_auc_score"
)
study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])
else:
study.optimize(objective, n_trials=n_trials)

# Obter melhores parâmetros
best_params = study.best_params.copy()
best_params.update({
'objective': 'binary:logistic',
'eval_metric': 'auc',
'random_state': 42,
'n_jobs': -1
})

logger.info(f"Melhor score encontrado: {study.best_value:.4f}")
logger.info(f"Melhores parâmetros: {best_params}")

# Salvar resultados da otimização
optimization_results = {
'best_params': best_params,
'best_score': study.best_value,
'n_trials': n_trials,
'study_name': study.study_name,
'optimization_date': datetime.now().isoformat()
}

# Salvar em arquivo para uso posterior
with open('optimization_results.json', 'w') as f:
json.dump(optimization_results, f, indent=2)

return best_params, study.best_value

def load_optimized_params(self, params_file='optimization_results.json'):
"""Carrega parâmetros otimizados de arquivo anterior"""
try:
if os.path.exists(params_file):
with open(params_file, 'r') as f:
optimization_data = json.load(f)
logger.info(f"Parâmetros otimizados carregados de {params_file}")
logger.info(f"Score anterior: {optimization_data['best_score']:.4f}")
return optimization_data['best_params'], optimization_data['best_score']
else:
logger.info("Arquivo de parâmetros otimizados não encontrado")
return None, None
except Exception as e:
logger.warning(f"Erro ao carregar parâmetros otimizados: {e}")
return None, None

def train_model(self, X, y, test_size=0.2, val_size=0.2, random_state=42, use_mlflow=True, 
use_optimized_params=True, n_trials=100, load_existing_params=False):
"""Treina o modelo XGBoost com integração MLflow e otimização bayesiana usando 3 folds"""
logger.info("Iniciando treinamento do modelo XGBoost com 3 folds (treino/validação/teste)...")

# Dividir dados em 3 folds: treino, validação e teste
# Primeiro split: separar teste (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Segundo split: separar treino e validação do restante (80%)
# Ajustar val_size para o dataset restante
adjusted_val_size = val_size / (1 - test_size) # val_size / 0.8
X_train, X_val, y_train, y_val = train_test_split(
X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
)

# Identificar variáveis contínuas para normalização
continuous_vars, discrete_vars = self.identify_continuous_variables(X_train)

# Aplicar normalização nos dados de validação e teste (usando parâmetros do treino)
if continuous_vars:
logger.info("Aplicando normalização nos dados de validação e teste...")
X_val = self.apply_normalization(X_val, continuous_vars, fit_scaler=False)
X_test = self.apply_normalization(X_test, continuous_vars, fit_scaler=False)

logger.info(f"3-Fold split:")
logger.info(f" Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
logger.info(f" Validação: {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
logger.info(f" Teste: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

# Otimizar hiperparâmetros se solicitado (usando treino + validação)
if use_optimized_params:
# Tentar carregar parâmetros existentes primeiro
if load_existing_params:
existing_params, existing_score = self.load_optimized_params()
if existing_params is not None:
logger.info("Usando parâmetros otimizados existentes...")
best_params, best_score = existing_params, existing_score
else:
logger.info("Executando nova otimização bayesiana...")
# Usar apenas treino + validação para otimização
X_opt = pd.concat([X_train, X_val], ignore_index=True)
y_opt = pd.concat([y_train, y_val], ignore_index=True)
best_params, best_score = self.optimize_hyperparameters(
X_opt, y_opt, n_trials=n_trials, val_size=val_size, use_mlflow=use_mlflow
)
else:
logger.info("Executando otimização bayesiana de hiperparâmetros...")
# Usar apenas treino + validação para otimização
X_opt = pd.concat([X_train, X_val], ignore_index=True)
y_opt = pd.concat([y_train, y_val], ignore_index=True)
best_params, best_score = self.optimize_hyperparameters(
X_opt, y_opt, n_trials=n_trials, val_size=val_size, use_mlflow=use_mlflow
)
logger.info(f"Otimização concluída. Melhor score: {best_score:.4f}")
else:
# Usar parâmetros padrão
best_params = {
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth': 6,
'learning_rate': 0.1,
'n_estimators': 200,
'subsample': 0.8,
'colsample_bytree': 0.8,
'random_state': random_state,
'n_jobs': -1
}
best_score = None

# Configurar MLflow
mlflow_config = get_mlflow_config()

# Iniciar run MLflow
if use_mlflow:
with mlflow_config.start_run(
run_name=f"xgboost-3fold-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
tags={
"model_type": "xgboost",
"dataset_size": len(X),
"features_count": len(X.columns),
"target_distribution": y.value_counts().to_dict(),
"optimization_used": str(use_optimized_params),
"n_trials": str(n_trials) if use_optimized_params else "default",
"split_type": "3fold"
}
) as run:
return self._train_with_mlflow(X_train, X_val, X_test, y_train, y_val, y_test, 
random_state, mlflow_config, run, best_params, best_score)
else:
return self._train_without_mlflow(X_train, X_val, X_test, y_train, y_val, y_test, 
random_state, best_params)

def _train_with_mlflow(self, X_train, X_val, X_test, y_train, y_val, y_test, 
random_state, mlflow_config, run, best_params, best_score):
"""Treinamento com MLflow tracking usando parâmetros otimizados e 3 folds"""
logger.info("Treinando modelo com MLflow tracking usando 3 folds...")

# Usar parâmetros otimizados
params = best_params.copy()
params['random_state'] = random_state

# Logar parâmetros
mlflow_config.log_model_params(params)

# Logar informações do dataset
dataset_info = {
'train_size': len(X_train),
'val_size': len(X_val),
'test_size': len(X_test),
'features_count': len(X_train.columns),
'target_positive_rate_train': y_train.mean(),
'target_positive_rate_val': y_val.mean(),
'target_positive_rate_test': y_test.mean()
}

# Adicionar score de otimização se disponível
if best_score is not None:
dataset_info['optimization_best_score'] = best_score

mlflow_config.log_model_params(dataset_info)

# Treinar modelo
self.model = xgb.XGBClassifier(**params)
self.model.fit(X_train, y_train)

# Avaliar modelo na validação
y_pred_val = self.model.predict(X_val)
y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred_proba_val)

# Avaliar modelo no teste (dados nunca vistos)
y_pred_test = self.model.predict(X_test)
y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_proba_test)

# Métricas para MLflow
metrics = {
'auc_validation': auc_val,
'auc_test': auc_test,
'auc_difference': abs(auc_test - auc_val)
}

# Logar métricas
mlflow_config.log_model_metrics(metrics)

# Logar importância das features
mlflow_config.log_feature_importance(
X_train.columns.tolist(), 
self.model.feature_importances_.tolist()
)

# Logar modelo
mlflow_config.log_xgboost_model(
self.model,
model_name="xgboost_model",
input_example=X_test.head(1)
)

# Logar artefatos
artifacts = {
"classification_report_val": "classification_report_val.txt",
"classification_report_test": "classification_report_test.txt",
"confusion_matrix_val": "confusion_matrix_val.txt",
"confusion_matrix_test": "confusion_matrix_test.txt"
}

# Salvar relatórios
with open("classification_report_val.txt", "w") as f:
f.write(classification_report(y_val, y_pred_val))

with open("classification_report_test.txt", "w") as f:
f.write(classification_report(y_test, y_pred_test))

with open("confusion_matrix_val.txt", "w") as f:
f.write(str(confusion_matrix(y_val, y_pred_val)))

with open("confusion_matrix_test.txt", "w") as f:
f.write(str(confusion_matrix(y_test, y_pred_test)))

mlflow_config.log_model_artifacts(artifacts)

# Limpar arquivos temporários
for file in artifacts.values():
if os.path.exists(file):
os.remove(file)

logger.info(f"AUC Validação: {auc_val:.4f}")
logger.info(f"AUC Teste: {auc_test:.4f}")
logger.info(f"Diferença: {abs(auc_test - auc_val):.4f}")
logger.info(f"MLflow Run ID: {run.info.run_id}")

self.is_trained = True

return {
'auc_validation': auc_val,
'auc_test': auc_test,
'auc_difference': abs(auc_test - auc_val),
'classification_report_val': classification_report(y_val, y_pred_val),
'classification_report_test': classification_report(y_test, y_pred_test),
'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_)),
'mlflow_run_id': run.info.run_id
}

def _train_without_mlflow(self, X_train, X_val, X_test, y_train, y_val, y_test, 
random_state, best_params):
"""Treinamento sem MLflow usando parâmetros otimizados e 3 folds"""
logger.info("Treinando modelo sem MLflow usando 3 folds...")

# Usar parâmetros otimizados
params = best_params.copy()
params['random_state'] = random_state

# Treinar modelo
self.model = xgb.XGBClassifier(**params)
self.model.fit(X_train, y_train)

# Avaliar modelo na validação
y_pred_val = self.model.predict(X_val)
y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred_proba_val)

# Avaliar modelo no teste
y_pred_test = self.model.predict(X_test)
y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_proba_test)

logger.info(f"AUC Validação: {auc_val:.4f}")
logger.info(f"AUC Teste: {auc_test:.4f}")
logger.info(f"Diferença: {abs(auc_test - auc_val):.4f}")

logger.info("\nClassification Report - Validação:")
logger.info(classification_report(y_val, y_pred_val))

logger.info("\nClassification Report - Teste:")
logger.info(classification_report(y_test, y_pred_test))

self.is_trained = True

return {
'auc_validation': auc_val,
'auc_test': auc_test,
'auc_difference': abs(auc_test - auc_val),
'classification_report_val': classification_report(y_val, y_pred_val),
'classification_report_test': classification_report(y_test, y_pred_test),
'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_))
}

def save_model(self, model_path='models/', register_in_mlflow=True):
"""Salva o modelo e preprocessadores"""
logger.info(f"Salvando modelo em: {model_path}")

# Criar diretório se não existir
os.makedirs(model_path, exist_ok=True)

# Salvar modelo
joblib.dump(self.model, f'{model_path}/xgboost_model.pkl')

# Salvar preprocessadores
joblib.dump(self.label_encoders, f'{model_path}/label_encoders.pkl')
joblib.dump(self.scaler, f'{model_path}/scaler.pkl')

# Salvar metadados
metadata = {
'feature_columns': self.feature_columns,
'categorical_columns': self.categorical_columns,
'model_type': 'XGBoost',
'training_date': datetime.now().isoformat(),
'is_trained': self.is_trained,
'normalization_applied': True,
'normalization_method': 'StandardScaler'
}

with open(f'{model_path}/model_metadata.json', 'w') as f:
json.dump(metadata, f, indent=2)

# Registrar no MLflow Model Registry se solicitado
if register_in_mlflow and self.is_trained:
try:
mlflow_config = get_mlflow_config()
registered_model = mlflow_config.register_model(
model_name="decision-recruitment-model",
stage="Production"
)
logger.info(f"Modelo registrado no MLflow: {registered_model.name} v{registered_model.version}")
except Exception as e:
logger.warning(f"Erro ao registrar modelo no MLflow: {e}")

logger.info("Modelo salvo com sucesso!")

def load_model(self, model_path='models/'):
"""Carrega modelo e preprocessadores"""
logger.info(f"Carregando modelo de: {model_path}")

self.model = joblib.load(f'{model_path}/xgboost_model.pkl')
self.label_encoders = joblib.load(f'{model_path}/label_encoders.pkl')
self.scaler = joblib.load(f'{model_path}/scaler.pkl')

with open(f'{model_path}/model_metadata.json', 'r') as f:
metadata = json.load(f)
self.feature_columns = metadata['feature_columns']
self.categorical_columns = metadata['categorical_columns']
self.is_trained = metadata['is_trained']

logger.info("Modelo carregado com sucesso!")

def predict(self, data):
"""Faz predição com novos dados"""
if not self.is_trained:
raise ValueError("Modelo não foi treinado ainda!")

# Preparar dados
if isinstance(data, dict):
data = pd.DataFrame([data])

# Aplicar feature engineering
# (implementar lógica similar ao treinamento)

# Preparar features para predição
valid_features = [col for col in self.feature_columns if col in data.columns]
X_pred = data[valid_features].fillna(0)

# Identificar variáveis contínuas e aplicar normalização
continuous_vars, _ = self.identify_continuous_variables(X_pred)
if continuous_vars:
logger.info("Aplicando normalização nos dados de predição...")
X_pred = self.apply_normalization(X_pred, continuous_vars, fit_scaler=False)

# Fazer predição
prediction = self.model.predict(X_pred)
probability = self.model.predict_proba(X_pred)[:, 1]

return {
'prediction': prediction[0],
'probability': probability[0],
'confidence': 'High' if probability[0] > 0.8 else 'Medium' if probability[0] > 0.5 else 'Low'
}

def main():
"""Função principal para treinamento"""
logger.info("=== INICIANDO PIPELINE DE TREINAMENTO ===")

# Inicializar modelo
model = DecisionRecruitmentModel()

# Carregar dados
model.load_data('dataset_preparado.csv')

# Feature engineering
model.advanced_feature_engineering()

# Preparar features
model.prepare_features()
model.encode_categorical_features()

# Preparar dados de treinamento
X, y = model.prepare_training_data()

# Logar informações do dataset no MLflow
try:
mlflow_config = get_mlflow_config()
mlflow_config.log_dataset_info('dataset_preparado.csv', 'contratado')
except Exception as e:
logger.warning(f"Erro ao logar informações do dataset: {e}")

# Treinar modelo com otimização bayesiana usando 3 folds (carrega parâmetros existentes se disponível)
results = model.train_model(X, y, test_size=0.2, val_size=0.2, use_mlflow=True, 
use_optimized_params=True, n_trials=50, load_existing_params=True)

# Salvar modelo
model.save_model(register_in_mlflow=True)

logger.info("=== PIPELINE CONCLUÍDA COM SUCESSO ===")

return model, results

if __name__ == "__main__":
model, results = main()
