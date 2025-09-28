#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAIN - Pipeline Completo de Treinamento com MLflow
Decision - Sistema de Match Candidato-Vaga
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import os
import logging
from datetime import datetime
import mlflow
import mlflow.xgboost
import optuna
from optuna.integration.mlflow import MLflowCallback
from mlflow_config import get_mlflow_config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecisionRecruitmentPipeline:
"""Pipeline completo de treinamento com MLflow"""

def __init__(self):
self.model = None
self.label_encoders = {}
self.scaler = StandardScaler()
self.feature_columns = []
self.categorical_columns = []
self.mlflow_config = get_mlflow_config()

def load_and_prepare_data(self):
"""Carrega e prepara os dados"""
logger.info("=== CARREGANDO E PREPARANDO DADOS ===")

# Carregar dados
df = pd.read_csv('dataset_preparado.csv')
logger.info(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")

# Feature engineering básico
df = self.basic_feature_engineering(df)

# Preparar features
self.prepare_features(df)

# Codificar features categóricas
self.encode_categorical_features(df)

# Preparar dados finais
X, y = self.prepare_training_data(df)

logger.info(f"Dataset final: {X.shape[0]} amostras, {X.shape[1]} features")
logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")

return X, y

def basic_feature_engineering(self, df):
"""Feature engineering básico"""
logger.info("Aplicando feature engineering básico...")

# Features de compatibilidade
nivel_mapping = {
'Júnior': 1, 'Pleno': 2, 'Sênior': 3, 'Especialista': 4, 'Líder': 5
}

idioma_mapping = {
'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4
}

# Compatibilidade de nível profissional
df['nivel_profissional_vaga_score'] = df['nivel_profissional_x'].map(nivel_mapping).fillna(0)
df['nivel_profissional_candidato_score'] = df['nivel_profissional_y'].map(nivel_mapping).fillna(0)
df['nivel_profissional_compatibility'] = 1 - abs(df['nivel_profissional_vaga_score'] - df['nivel_profissional_candidato_score']) / 4

# Compatibilidade de inglês
df['nivel_ingles_vaga_score'] = df['nivel_ingles_x'].map(idioma_mapping).fillna(0)
df['nivel_ingles_candidato_score'] = df['nivel_ingles_y'].map(idioma_mapping).fillna(0)
df['nivel_ingles_compatibility'] = 1 - abs(df['nivel_ingles_vaga_score'] - df['nivel_ingles_candidato_score']) / 4

# Compatibilidade de espanhol
df['nivel_espanhol_vaga_score'] = df['nivel_espanhol_x'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_candidato_score'] = df['nivel_espanhol_y'].map(idioma_mapping).fillna(0)
df['nivel_espanhol_compatibility'] = 1 - abs(df['nivel_espanhol_vaga_score'] - df['nivel_espanhol_candidato_score']) / 4

# Features de texto
df['cv_length'] = df['cv_pt'].str.len().fillna(0)
df['cv_has_technical_keywords'] = df['cv_pt'].str.contains(
'python|java|javascript|sql|aws|azure|docker|kubernetes|react|angular|node', 
case=False, na=False
).astype(int)

df['cv_has_certifications'] = df['cv_pt'].str.contains(
'certificação|certificado|certified|aws|microsoft|google|oracle', 
case=False, na=False
).astype(int)

# Features básicas
df['is_sap_vaga'] = (df['is_sap_vaga'] == 1).astype(int)
df['is_pcd'] = (df['is_pcd'] == 1).astype(int)
df['is_sp'] = df['cidade'].str.contains('São Paulo', na=False).astype(int)

# Features de match
df['nivel_profissional_match'] = (df['nivel_profissional_x'] == df['nivel_profissional_y']).astype(int)
df['nivel_ingles_match'] = (df['nivel_ingles_x'] == df['nivel_ingles_y']).astype(int)
df['nivel_espanhol_match'] = (df['nivel_espanhol_x'] == df['nivel_espanhol_y']).astype(int)

return df

def prepare_features(self, df):
"""Prepara lista de features"""
# Features numéricas
self.feature_columns = [
'is_sap_vaga', 'nivel_profissional_match', 'nivel_academico_match',
'nivel_ingles_match', 'nivel_espanhol_match', 'cv_length', 'has_cv_en',
'dias_entre_requisicao_candidatura', 'is_pcd', 'is_sp', 'remuneracao_numeric',
'nivel_profissional_compatibility', 'nivel_ingles_compatibility', 
'nivel_espanhol_compatibility', 'cv_has_technical_keywords',
'cv_has_certifications'
]

# Features categóricas
self.categorical_columns = [
'cliente', 'nivel_profissional_x', 'nivel_academico_x', 'nivel_ingles_x', 
'nivel_espanhol_x', 'area_atuacao', 'cidade', 'tipo_contratacao',
'titulo_profissional', 'nivel_profissional_y', 'nivel_academico_y',
'nivel_ingles_y', 'nivel_espanhol_y', 'recrutador'
]

# Filtrar colunas que existem
self.feature_columns = [col for col in self.feature_columns if col in df.columns]
self.categorical_columns = [col for col in self.categorical_columns if col in df.columns]

logger.info(f"Features numéricas: {len(self.feature_columns)}")
logger.info(f"Features categóricas: {len(self.categorical_columns)}")

def encode_categorical_features(self, df):
"""Codifica features categóricas"""
logger.info("Codificando features categóricas...")

for col in self.categorical_columns:
if col in df.columns:
le = LabelEncoder()
df[col] = df[col].astype(str).fillna('Unknown')
df[f'{col}_encoded'] = le.fit_transform(df[col])
self.label_encoders[col] = le
self.feature_columns.append(f'{col}_encoded')

logger.info("Features categóricas codificadas!")

def prepare_training_data(self, df):
"""Prepara dados para treinamento"""
# Filtrar features válidas
valid_features = [col for col in self.feature_columns if col in df.columns]

# Remover linhas com target nulo
df_clean = df.dropna(subset=['contratado'])

# Separar features e target
X = df_clean[valid_features].fillna(0)
y = df_clean['contratado']

# Identificar variáveis contínuas
continuous_vars = self.identify_continuous_variables(X)

# Aplicar normalização
if continuous_vars:
X[continuous_vars] = self.scaler.fit_transform(X[continuous_vars])

return X, y

def identify_continuous_variables(self, X):
"""Identifica variáveis contínuas"""
continuous_vars = []
for col in X.columns:
sample_values = X[col].dropna().head(100)
for value in sample_values:
if pd.notna(value) and isinstance(value, (int, float)):
if value != int(value):
continuous_vars.append(col)
break
return continuous_vars

def optimize_hyperparameters(self, X, y, n_trials=50):
"""Otimização bayesiana de hiperparâmetros"""
logger.info(f"=== INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS ({n_trials} trials) ===")

# Dividir dados
X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

def objective(trial):
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

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_pred_proba)

return auc_score

# Configurar estudo
study = optuna.create_study(
direction='maximize',
study_name='xgboost_hyperparameter_optimization'
)

# Executar otimização
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

# Salvar resultados
optimization_results = {
'best_params': best_params,
'best_score': study.best_value,
'n_trials': n_trials,
'optimization_date': datetime.now().isoformat()
}

with open('optimization_results.json', 'w') as f:
json.dump(optimization_results, f, indent=2)

return best_params, study.best_value

def train_final_model(self, X, y, best_params):
"""Treina modelo final com parâmetros otimizados"""
logger.info("=== TREINANDO MODELO FINAL ===")

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

# Iniciar run MLflow
with self.mlflow_config.start_run(
run_name=f"final-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
tags={
"model_type": "xgboost",
"dataset_size": len(X),
"features_count": len(X.columns),
"optimization_used": "True"
}
) as run:

# Logar parâmetros
self.mlflow_config.log_model_params(best_params)

# Logar informações do dataset
dataset_info = {
'train_size': len(X_train),
'test_size': len(X_test),
'features_count': len(X_train.columns),
'target_positive_rate_train': y_train.mean(),
'target_positive_rate_test': y_test.mean()
}
self.mlflow_config.log_model_params(dataset_info)

# Treinar modelo
self.model = xgb.XGBClassifier(**best_params)
self.model.fit(X_train, y_train)

# Avaliar modelo
y_pred = self.model.predict(X_test)
y_pred_proba = self.model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

# Validação cruzada
cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Métricas
metrics = {
'auc_score': auc_score,
'cv_auc_mean': cv_mean,
'cv_auc_std': cv_std,
'cv_auc_min': cv_scores.min(),
'cv_auc_max': cv_scores.max()
}

# Logar métricas
self.mlflow_config.log_model_metrics(metrics)

# Logar importância das features
self.mlflow_config.log_feature_importance(
X_train.columns.tolist(), 
self.model.feature_importances_.tolist()
)

# Logar modelo
self.mlflow_config.log_xgboost_model(
self.model,
model_name="xgboost_model",
input_example=X_test.head(1)
)

logger.info(f"AUC Score: {auc_score:.4f}")
logger.info(f"CV AUC Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
logger.info(f"MLflow Run ID: {run.info.run_id}")

return {
'auc_score': auc_score,
'cv_auc_mean': cv_mean,
'cv_auc_std': cv_std,
'mlflow_run_id': run.info.run_id
}

def save_model(self):
"""Salva modelo e preprocessadores"""
logger.info("=== SALVANDO MODELO ===")

# Criar diretório
os.makedirs('models/', exist_ok=True)

# Salvar modelo
joblib.dump(self.model, 'models/xgboost_model.pkl')
joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
joblib.dump(self.scaler, 'models/scaler.pkl')

# Salvar metadados
metadata = {
'feature_columns': self.feature_columns,
'categorical_columns': self.categorical_columns,
'model_type': 'XGBoost',
'training_date': datetime.now().isoformat(),
'is_trained': True
}

with open('models/model_metadata.json', 'w') as f:
json.dump(metadata, f, indent=2)

logger.info("Modelo salvo com sucesso!")

def register_model_in_mlflow(self):
"""Registra modelo no MLflow Model Registry"""
logger.info("=== REGISTRANDO MODELO NO MLFLOW ===")

try:
registered_model = self.mlflow_config.register_model(
model_name="decision-recruitment-model",
stage="Production"
)
logger.info(f"Modelo registrado: {registered_model.name} v{registered_model.version}")
return True
except Exception as e:
logger.error(f"Erro ao registrar modelo: {e}")
return False

def main():
"""Função principal"""
logger.info(" INICIANDO PIPELINE COMPLETO DE TREINAMENTO")

try:
# Inicializar pipeline
pipeline = DecisionRecruitmentPipeline()

# 1. Carregar e preparar dados
X, y = pipeline.load_and_prepare_data()

# 2. Otimizar hiperparâmetros
best_params, best_score = pipeline.optimize_hyperparameters(X, y, n_trials=30)

# 3. Treinar modelo final
results = pipeline.train_final_model(X, y, best_params)

# 4. Salvar modelo
pipeline.save_model()

# 5. Registrar no MLflow
registry_success = pipeline.register_model_in_mlflow()

# Resumo final
logger.info("\n PIPELINE CONCLUÍDO COM SUCESSO!")
logger.info(f" Melhor AUC Score: {results['auc_score']:.4f}")
logger.info(f" CV AUC Score: {results['cv_auc_mean']:.4f} (+/- {results['cv_auc_std']*2:.4f})")
logger.info(f" MLflow Run ID: {results['mlflow_run_id']}")
logger.info(f" Modelo salvo em: models/")
logger.info(f" Modelo registrado: {'Sim' if registry_success else 'Não'}")

logger.info("\n Para visualizar os resultados:")
logger.info(" mlflow ui --backend-store-uri file:./mlruns")
logger.info(" Acesse: http://localhost:5000")

return pipeline, results

except Exception as e:
logger.error(f" Erro no pipeline: {e}")
raise

if __name__ == "__main__":
pipeline, results = main()
