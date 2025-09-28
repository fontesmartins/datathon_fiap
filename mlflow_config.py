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
            tracking_uri: URI do tracking store
            registry_uri: URI do model registry
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        
        # Configurar MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Configura MLflow tracking e registry"""
        # Configurar tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Configurar registry URI se fornecido
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        # Criar ou obter experimento
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags={
                        "project": "decision-recruitment-ai",
                        "description": "Sistema de match candidato-vaga usando XGBoost",
                        "created_at": datetime.now().isoformat()
                    }
                )
            else:
                self.experiment_id = self.experiment.experiment_id
                
            # Definir experimento ativo
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"MLflow configurado - Experimento: {self.experiment_name} (ID: {self.experiment_id})")
            
        except Exception as e:
            logger.error(f"Erro ao configurar MLflow: {e}")
            raise
    
    def start_run(self, run_name=None, tags=None):
        """Inicia um novo run MLflow"""
        return mlflow.start_run(
            run_name=run_name,
            tags=tags or {}
        )
    
    def log_model_params(self, params):
        """Loga parâmetros do modelo"""
        if isinstance(params, dict):
            mlflow.log_params(params)
        else:
            mlflow.log_param("params", str(params))
    
    def log_model_metrics(self, metrics):
        """Loga métricas do modelo"""
        if isinstance(metrics, dict):
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metric("metric", metrics)
    
    def log_feature_importance(self, feature_names, importance_values):
        """Loga importância das features"""
        import pandas as pd
        
        # Criar DataFrame com importância das features
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Logar como artefato
        os.makedirs("feature_importance", exist_ok=True)
        importance_path = "feature_importance/feature_importance.csv"
        feature_importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Logar métricas individuais das top features
        for i, row in feature_importance_df.head(10).iterrows():
            metric_name = f"feature_importance_{row['feature']}"
            mlflow.log_metric(metric_name, row['importance'])
    
    def log_xgboost_model(self, model, model_name="xgboost_model", input_example=None):
        """Loga modelo XGBoost"""
        try:
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=model_name,
                input_example=input_example,
                registered_model_name="decision-recruitment-model"
            )
            logger.info(f"Modelo XGBoost logado: {model_name}")
        except Exception as e:
            logger.error(f"Erro ao logar modelo XGBoost: {e}")
            raise
    
    def log_sklearn_model(self, model, model_name="sklearn_model", input_example=None):
        """Loga modelo Scikit-learn"""
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                input_example=input_example,
                registered_model_name="decision-recruitment-model"
            )
            logger.info(f"Modelo Scikit-learn logado: {model_name}")
        except Exception as e:
            logger.error(f"Erro ao logar modelo Scikit-learn: {e}")
            raise
    
    def register_model(self, model_name="decision-recruitment-model", stage="Production"):
        """Registra modelo no Model Registry"""
        try:
            # Obter run atual
            current_run = mlflow.active_run()
            if current_run is None:
                raise ValueError("Nenhum run ativo encontrado")
            
            # Registrar modelo
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{current_run.info.run_id}/xgboost_model",
                name=model_name
            )
            
            # Adicionar tags ao modelo
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="stage",
                value=stage
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="registered_at",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Modelo registrado: {model_name} v{registered_model.version}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Erro ao registrar modelo: {e}")
            raise
    
    def get_model_info(self, model_name="decision-recruitment-model", stage="Production"):
        """Obtém informações do modelo registrado"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Buscar modelo por stage
            if stage:
                model_version = client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )[0]
            else:
                model_version = client.get_latest_versions(model_name)[0]
            
            # Obter informações do run
            run = client.get_run(model_version.run_id)
            
            return {
                "model_name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "creation_timestamp": model_version.creation_timestamp,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter informações do modelo: {e}")
            return None
    
    def list_models(self):
        """Lista todos os modelos registrados"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            
            model_list = []
            for model in models:
                model_info = {
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "creation_timestamp": version.creation_timestamp
                        }
                        for version in model.latest_versions
                    ]
                }
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []
    
    def transition_model_stage(self, model_name, version, stage):
        """Transiciona modelo para um novo stage"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Modelo {model_name} v{version} transicionado para {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao transicionar modelo: {e}")
            return False

# Instância global da configuração
mlflow_config = MLflowConfig()

def get_mlflow_config():
    """Retorna instância global da configuração MLflow"""
    return mlflow_config
