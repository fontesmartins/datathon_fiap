#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Experimentos MLflow para Decision Recruitment AI
Comparação de diferentes configurações de modelo e hiperparâmetros
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import logging
from datetime import datetime
import itertools
from model_pipeline import DecisionRecruitmentModel
from mlflow_config import get_mlflow_config

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLflowExperimentRunner:
    """Runner para experimentos MLflow"""
    
    def __init__(self):
        self.mlflow_config = get_mlflow_config()
        self.model = DecisionRecruitmentModel()
        
    def load_and_prepare_data(self):
        """Carrega e prepara dados para experimentos"""
        logger.info("Carregando e preparando dados...")
        
        # Carregar dados
        self.model.load_data('dataset_preparado.csv')
        
        # Feature engineering
        self.model.advanced_feature_engineering()
        
        # Preparar features
        self.model.prepare_features()
        self.model.encode_categorical_features()
        
        # Preparar dados de treinamento
        X, y = self.model.prepare_training_data()
        
        logger.info(f"Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
        return X, y
    
    def run_hyperparameter_experiment(self, X, y):
        """Executa experimento de hiperparâmetros"""
        logger.info("=== INICIANDO EXPERIMENTO DE HIPERPARÂMETROS ===")
        
        # Grid de hiperparâmetros para testar
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Gerar combinações (amostra para não sobrecarregar)
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        # Limitar a 20 combinações para demonstração
        if len(param_combinations) > 20:
            param_combinations = param_combinations[:20]
        
        logger.info(f"Testando {len(param_combinations)} combinações de hiperparâmetros")
        
        best_score = 0
        best_params = None
        best_run_id = None
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            # Adicionar parâmetros fixos
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1
            })
            
            logger.info(f"Executando experimento {i+1}/{len(param_combinations)}: {params}")
            
            try:
                with self.mlflow_config.start_run(
                    run_name=f"hyperparameter-exp-{i+1}",
                    tags={
                        "experiment_type": "hyperparameter_tuning",
                        "model_type": "xgboost",
                        "combination_id": i+1
                    }
                ) as run:
                    
                    # Logar parâmetros
                    self.mlflow_config.log_model_params(params)
                    
                    # Dividir dados
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Treinar modelo
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    # Avaliar modelo
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    # Validação cruzada
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
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
                    
                    # Logar modelo se for o melhor até agora
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_params = params.copy()
                        best_run_id = run.info.run_id
                        
                        # Logar modelo
                        self.mlflow_config.log_xgboost_model(
                            model,
                            model_name="best_hyperparameter_model",
                            input_example=X_test.head(1)
                        )
                    
                    logger.info(f"   AUC: {auc_score:.4f}, CV AUC: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
                    
            except Exception as e:
                logger.error(f"Erro no experimento {i+1}: {e}")
                continue
        
        logger.info(f"=== MELHOR RESULTADO ===")
        logger.info(f"CV AUC Score: {best_score:.4f}")
        logger.info(f"Melhores parâmetros: {best_params}")
        logger.info(f"Run ID: {best_run_id}")
        
        return best_params, best_score, best_run_id
    
    def run_model_comparison_experiment(self, X, y):
        """Executa experimento comparando diferentes algoritmos"""
        logger.info("=== INICIANDO EXPERIMENTO DE COMPARAÇÃO DE MODELOS ===")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Configurações de modelos
        models_config = {
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000
                }
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Testando modelo: {model_name}")
            
            try:
                with self.mlflow_config.start_run(
                    run_name=f"model-comparison-{model_name}",
                    tags={
                        "experiment_type": "model_comparison",
                        "model_type": model_name
                    }
                ) as run:
                    
                    # Logar parâmetros
                    self.mlflow_config.log_model_params(config['params'])
                    
                    # Dividir dados
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Treinar modelo
                    model = config['model'](**config['params'])
                    model.fit(X_train, y_train)
                    
                    # Avaliar modelo
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    # Validação cruzada
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
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
                    
                    # Logar modelo
                    if model_name == 'xgboost':
                        self.mlflow_config.log_xgboost_model(
                            model,
                            model_name=f"{model_name}_model",
                            input_example=X_test.head(1)
                        )
                    else:
                        self.mlflow_config.log_sklearn_model(
                            model,
                            model_name=f"{model_name}_model",
                            input_example=X_test.head(1)
                        )
                    
                    results[model_name] = {
                        'auc_score': auc_score,
                        'cv_auc_mean': cv_mean,
                        'cv_auc_std': cv_std,
                        'run_id': run.info.run_id
                    }
                    
                    logger.info(f"   {model_name}: AUC {auc_score:.4f}, CV AUC {cv_mean:.4f} (+/- {cv_std*2:.4f})")
                    
            except Exception as e:
                logger.error(f"Erro no modelo {model_name}: {e}")
                continue
        
        # Encontrar melhor modelo
        best_model = max(results.items(), key=lambda x: x[1]['cv_auc_mean'])
        logger.info(f"=== MELHOR MODELO ===")
        logger.info(f"Modelo: {best_model[0]}")
        logger.info(f"CV AUC Score: {best_model[1]['cv_auc_mean']:.4f}")
        logger.info(f"Run ID: {best_model[1]['run_id']}")
        
        return results, best_model
    
    def run_feature_importance_experiment(self, X, y):
        """Executa experimento analisando importância das features"""
        logger.info("=== INICIANDO EXPERIMENTO DE IMPORTÂNCIA DAS FEATURES ===")
        
        # Parâmetros otimizados
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        try:
            with self.mlflow_config.start_run(
                run_name="feature-importance-analysis",
                tags={
                    "experiment_type": "feature_analysis",
                    "model_type": "xgboost"
                }
            ) as run:
                
                # Logar parâmetros
                self.mlflow_config.log_model_params(params)
                
                # Dividir dados
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Treinar modelo
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                
                # Avaliar modelo
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Validação cruzada
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Métricas
                metrics = {
                    'auc_score': auc_score,
                    'cv_auc_mean': cv_mean,
                    'cv_auc_std': cv_std
                }
                
                # Logar métricas
                self.mlflow_config.log_model_metrics(metrics)
                
                # Logar importância das features
                self.mlflow_config.log_feature_importance(
                    X.columns.tolist(),
                    model.feature_importances_.tolist()
                )
                
                # Logar modelo
                self.mlflow_config.log_xgboost_model(
                    model,
                    model_name="feature_analysis_model",
                    input_example=X_test.head(1)
                )
                
                logger.info(f"Análise de features concluída: AUC {auc_score:.4f}, CV AUC {cv_mean:.4f}")
                
                return {
                    'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                    'auc_score': auc_score,
                    'cv_auc_mean': cv_mean,
                    'run_id': run.info.run_id
                }
                
        except Exception as e:
            logger.error(f"Erro na análise de features: {e}")
            return None
    
    def run_all_experiments(self):
        """Executa todos os experimentos"""
        logger.info("=== INICIANDO SUITE COMPLETA DE EXPERIMENTOS ===")
        
        # Carregar e preparar dados
        X, y = self.load_and_prepare_data()
        
        # Logar informações do dataset
        self.mlflow_config.log_dataset_info('dataset_preparado.csv', 'contratado')
        
        # Executar experimentos
        results = {}
        
        # 1. Experimento de hiperparâmetros
        logger.info("\n" + "="*50)
        hyperparams_results = self.run_hyperparameter_experiment(X, y)
        results['hyperparameter_tuning'] = hyperparams_results
        
        # 2. Comparação de modelos
        logger.info("\n" + "="*50)
        model_comparison_results = self.run_model_comparison_experiment(X, y)
        results['model_comparison'] = model_comparison_results
        
        # 3. Análise de features
        logger.info("\n" + "="*50)
        feature_analysis_results = self.run_feature_importance_experiment(X, y)
        results['feature_analysis'] = feature_analysis_results
        
        logger.info("\n=== EXPERIMENTOS CONCLUÍDOS ===")
        logger.info("Acesse o MLflow UI para visualizar os resultados:")
        logger.info("mlflow ui --backend-store-uri file:./mlruns")
        
        return results

def main():
    """Função principal para executar experimentos"""
    try:
        runner = MLflowExperimentRunner()
        results = runner.run_all_experiments()
        
        print("\n🎉 Experimentos concluídos com sucesso!")
        print("📊 Para visualizar os resultados, execute:")
        print("   mlflow ui --backend-store-uri file:./mlruns")
        print("   Acesse: http://localhost:5000")
        
        return results
        
    except Exception as e:
        logger.error(f"Erro na execução dos experimentos: {e}")
        raise

if __name__ == "__main__":
    results = main()
