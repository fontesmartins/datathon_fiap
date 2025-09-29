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
label_encoders = None
scaler = None
model_metadata = None

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
    vaga_sap: str = Field(default="Não", description="Vaga SAP")
    competencia_tecnicas: str = Field(default="", description="Competências técnicas")
    tipo_contratacao: str = Field(..., description="Tipo de contratação")

class PredictionRequest(BaseModel):
    """Modelo para requisição de predição"""
    candidate: CandidateData
    job: JobData

class PredictionResponse(BaseModel):
    """Modelo para resposta de predição"""
    status: str
    prediction: int
    probability: float
    confidence_score: float
    recommendation: str
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Modelo para predição em lote"""
    candidates: List[CandidateData]
    job: JobData

class BatchPredictionResponse(BaseModel):
    """Modelo para resposta de predição em lote"""
    status: str
    predictions: List[Dict[str, Any]]
    model_version: str
    timestamp: str

def load_model():
    """Carrega modelo e preprocessadores"""
    global model, label_encoders, scaler, model_metadata
    
    try:
        # Verificar se os arquivos existem
        model_path = "models/xgboost_model.pkl"
        encoders_path = "models/label_encoders.pkl"
        scaler_path = "models/scaler.pkl"
        metadata_path = "models/model_metadata.json"
        
        if not all(os.path.exists(path) for path in [model_path, encoders_path, scaler_path, metadata_path]):
            raise FileNotFoundError("Arquivos do modelo não encontrados")
        
        # Carregar modelo e preprocessadores
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        scaler = joblib.load(scaler_path)
        
        # Carregar metadados
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        logger.info("Modelo carregado com sucesso!")
        logger.info(f"Tipo do modelo: {model_metadata.get('model_type', 'Desconhecido')}")
        logger.info(f"Data de treinamento: {model_metadata.get('training_date', 'Desconhecida')}")
        logger.info(f"Número de features: {len(model_metadata.get('feature_columns', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return False

def prepare_features(candidate_data: CandidateData, job_data: JobData) -> np.ndarray:
    """Prepara features para predição"""
    try:
        # Criar DataFrame com os dados
        data = {
            'nivel_profissional_x': [job_data.nivel_profissional_vaga],
            'nivel_academico_x': ['Ensino Superior Completo'],  # Valor padrão
            'nivel_ingles_x': [job_data.nivel_ingles_vaga],
            'nivel_espanhol_x': [job_data.nivel_espanhol_vaga],
            'area_atuacao': ['TI - Desenvolvimento/Programação'],  # Valor padrão
            'cidade': [candidate_data.estado],
            'tipo_contratacao': [job_data.tipo_contratacao],
            'titulo_profissional': [job_data.titulo_vaga],
            'nivel_profissional_y': [candidate_data.nivel_profissional_candidato],
            'nivel_academico_y': ['Ensino Superior Completo'],  # Valor padrão
            'nivel_ingles_y': [candidate_data.nivel_ingles_candidato],
            'nivel_espanhol_y': [candidate_data.nivel_espanhol_candidato],
            'is_sap_vaga': [1 if job_data.vaga_sap.lower() == 'sim' else 0],
            'cv_pt': [candidate_data.cv_text],
            'remuneracao_numeric': [candidate_data.remuneracao or 0],
            'is_pcd': [1 if candidate_data.pcd.lower() == 'sim' else 0],
            'has_cv_en': [1 if len(candidate_data.cv_text) > 0 else 0],
            'dias_entre_requisicao_candidatura': [0]  # Valor padrão
        }
        
        df = pd.DataFrame(data)
        
        # Feature engineering básico
        df = apply_feature_engineering(df)
        
        # Codificar features categóricas
        feature_columns = []
        
        # Features numéricas
        numeric_features = [
            'is_sap_vaga', 'remuneracao_numeric', 'is_pcd', 'has_cv_en',
            'dias_entre_requisicao_candidatura', 'nivel_profissional_compatibility',
            'nivel_ingles_compatibility', 'nivel_espanhol_compatibility',
            'cv_length', 'cv_has_technical_keywords', 'cv_has_certifications',
            'nivel_profissional_match', 'nivel_ingles_match', 'nivel_espanhol_match',
            'is_sp'
        ]
        
        # Filtrar features que existem
        numeric_features = [f for f in numeric_features if f in df.columns]
        
        # Codificar features categóricas (removido cliente e recrutador)
        for col in ['nivel_profissional_x', 'nivel_academico_x', 
                   'nivel_ingles_x', 'nivel_espanhol_x', 'area_atuacao', 
                   'cidade', 'tipo_contratacao', 'titulo_profissional',
                   'nivel_profissional_y', 'nivel_academico_y', 
                   'nivel_ingles_y', 'nivel_espanhol_y']:
            if col in df.columns and col in label_encoders:
                df[col] = df[col].astype(str).fillna('Unknown')
                df[f'{col}_encoded'] = label_encoders[col].transform(df[col])
                feature_columns.append(f'{col}_encoded')
        
        # Adicionar features numéricas
        feature_columns.extend(numeric_features)
        
        # Filtrar features válidas
        valid_features = [f for f in feature_columns if f in df.columns]
        
        # Preparar dados finais
        X = df[valid_features].fillna(0)
        
        # Aplicar normalização se necessário
        continuous_vars = ['remuneracao_numeric', 'cv_length', 'dias_entre_requisicao_candidatura']
        continuous_vars = [f for f in continuous_vars if f in X.columns]
        
        if continuous_vars:
            X[continuous_vars] = scaler.transform(X[continuous_vars])
        
        return X.values
        
    except Exception as e:
        logger.error(f"Erro ao preparar features: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao preparar features: {str(e)}")

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica feature engineering"""
    try:
        # Mapeamentos
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
        
    except Exception as e:
        logger.error(f"Erro no feature engineering: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização da API"""
    logger.info("Iniciando Decision Recruitment AI API...")
    
    if not load_model():
        logger.error("Falha ao carregar modelo. API não pode ser iniciada.")
        raise RuntimeError("Modelo não pôde ser carregado")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check da API"""
    try:
        # Verificar se todos os componentes estão carregados
        components_status = {
            "model": model is not None,
            "label_encoders": label_encoders is not None,
            "scaler": scaler is not None,
            "model_metadata": model_metadata is not None
        }
        
        all_loaded = all(components_status.values())
        
        if not all_loaded:
            logger.warning(f"Componentes não carregados: {components_status}")
            raise HTTPException(
                status_code=503, 
                detail=f"Service unavailable - Components not loaded: {components_status}"
            )
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat(),
            "model_version": model_metadata.get('training_date', 'unknown') if model_metadata else 'unknown',
            "components": components_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_match(request: PredictionRequest):
    """Predição de match candidato-vaga"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        # Preparar features
        X = prepare_features(request.candidate, request.job)
        
        # Fazer predição
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Calcular confidence score
        confidence_score = abs(probability - 0.5) * 2
        
        # Determinar recomendação
        if prediction == 1 and probability > 0.7:
            recommendation = "Alta recomendação para contratação"
        elif prediction == 1 and probability > 0.5:
            recommendation = "Recomendação moderada para contratação"
        elif prediction == 0 and probability < 0.3:
            recommendation = "Não recomendado para contratação"
        else:
            recommendation = "Avaliação adicional necessária"
        
        return PredictionResponse(
            status="success",
            prediction=int(prediction),
            probability=float(probability),
            confidence_score=float(confidence_score),
            recommendation=recommendation,
            model_version=model_metadata.get('training_date', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predição em lote de múltiplos candidatos"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        predictions = []
        
        for i, candidate in enumerate(request.candidates):
            try:
                # Preparar features
                X = prepare_features(candidate, request.job)
                
                # Fazer predição
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0][1]
                confidence_score = abs(probability - 0.5) * 2
                
                # Determinar recomendação
                if prediction == 1 and probability > 0.7:
                    recommendation = "Alta recomendação"
                elif prediction == 1 and probability > 0.5:
                    recommendation = "Recomendação moderada"
                elif prediction == 0 and probability < 0.3:
                    recommendation = "Não recomendado"
                else:
                    recommendation = "Avaliação adicional necessária"
                
                predictions.append({
                    "candidate_name": candidate.nome,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "confidence_score": float(confidence_score),
                    "recommendation": recommendation
                })
                
            except Exception as e:
                logger.error(f"Erro na predição do candidato {i}: {e}")
                predictions.append({
                    "candidate_name": candidate.nome,
                    "prediction": -1,
                    "probability": 0.0,
                    "confidence_score": 0.0,
                    "recommendation": f"Erro: {str(e)}"
                })
        
        return BatchPredictionResponse(
            status="success",
            predictions=predictions,
            model_version=model_metadata.get('training_date', 'unknown') if model_metadata else 'unknown',
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")

@app.get("/model_info", tags=["Model"])
async def get_model_info():
    """Informações sobre o modelo"""
    if model is None or model_metadata is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    
    return {
        "model_type": model_metadata.get('model_type', 'Desconhecido'),
        "training_date": model_metadata.get('training_date', 'Desconhecida'),
        "is_trained": model_metadata.get('is_trained', False),
        "features_count": len(model_metadata.get('feature_columns', [])),
        "feature_columns": model_metadata.get('feature_columns', []),
        "categorical_columns": model_metadata.get('categorical_columns', [])
    }

@app.get("/feature_importance", tags=["Model"])
async def get_feature_importance():
    """Importância das features do modelo"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        # Obter importância das features
        feature_importance = model.feature_importances_
        feature_names = model_metadata.get('feature_columns', [])
        
        # Criar lista de features com importância
        importance_data = []
        for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
            importance_data.append({
                "feature": name,
                "importance": float(importance),
                "rank": i + 1
            })
        
        # Ordenar por importância
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "feature_importance": importance_data[:20],  # Top 20 features
            "total_features": len(feature_names)
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter importância das features: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter importância das features: {str(e)}")

@app.get("/debug", tags=["Debug"])
async def debug_info():
    """Informações de debug para troubleshooting"""
    try:
        import os
        
        debug_info = {
            "model_loaded": model is not None,
            "label_encoders_loaded": label_encoders is not None,
            "scaler_loaded": scaler is not None,
            "model_metadata_loaded": model_metadata is not None,
            "working_directory": os.getcwd(),
            "files_in_models_dir": [],
            "files_in_root": []
        }
        
        # Verificar arquivos
        if os.path.exists('models/'):
            debug_info["files_in_models_dir"] = os.listdir('models/')
        
        debug_info["files_in_root"] = [f for f in os.listdir('.') if f.endswith(('.pkl', '.json', '.csv'))]
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug_features", tags=["Debug"])
async def debug_features():
    """Testa a geração de features"""
    try:
        from fastapi_app import CandidateData, JobData
        
        # Dados de teste
        candidate_data = CandidateData(
            nome="Teste",
            nivel_profissional_candidato="Sênior",
            nivel_ingles_candidato="Avançado",
            nivel_espanhol_candidato="Intermediário",
            cv_text="Python developer",
            pcd="Não",
            remuneracao=8000,
            estado="São Paulo"
        )
        
        job_data = JobData(
            titulo_vaga="Analista Desenvolvedor",
            nivel_profissional_vaga="Sênior",
            nivel_ingles_vaga="Avançado",
            nivel_espanhol_vaga="Intermediário",
            vaga_sap="Não",
            competencia_tecnicas="Python",
            tipo_contratacao="CLT Full"
        )
        
        # Testar prepare_features
        X = prepare_features(candidate_data, job_data)
        
        return {
            "features_shape": X.shape,
            "features_columns": list(X.columns) if hasattr(X, 'columns') else "No columns",
            "compatibility_features": [
                "nivel_profissional_compatibility" in (X.columns if hasattr(X, 'columns') else []),
                "nivel_ingles_compatibility" in (X.columns if hasattr(X, 'columns') else []),
                "nivel_espanhol_compatibility" in (X.columns if hasattr(X, 'columns') else [])
            ]
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}

@app.get("/mlflow_info", tags=["MLflow"])
async def get_mlflow_info():
    """Informações sobre experimentos MLflow"""
    try:
        import mlflow
        
        # Tentar conectar ao MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Obter informações básicas
        experiments = mlflow.search_experiments()
        
        experiment_info = []
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            experiment_info.append({
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.name,
                "runs_count": len(runs),
                "creation_time": exp.creation_time.isoformat() if exp.creation_time else None
            })
        
        return {
            "mlflow_tracking_uri": "file:./mlruns",
            "experiments": experiment_info,
            "status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Erro ao conectar ao MLflow: {e}")
        return {
            "mlflow_tracking_uri": "file:./mlruns",
            "experiments": [],
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
