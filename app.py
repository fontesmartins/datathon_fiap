#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Flask para Modelo Preditivo de Recrutamento
Decision - Sistema de Match Candidato-Vaga
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import os
from model_pipeline import DecisionRecruitmentModel

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Carregar modelo globalmente
model = None

def load_model():
    """Carrega o modelo treinado"""
    global model
    try:
        model = DecisionRecruitmentModel()
        model.load_model('models/')
        logger.info("Modelo carregado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocessa dados de entrada para predição"""
    try:
        # Converter para DataFrame se necessário
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Aplicar feature engineering básico
        df = apply_basic_feature_engineering(df)
        
        # Selecionar apenas features necessárias
        feature_columns = model.feature_columns
        
        # Filtrar colunas que existem
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Preencher valores faltantes
        df_features = df[available_features].fillna(0)
        
        return df_features
        
    except Exception as e:
        logger.error(f"Erro no preprocessamento: {str(e)}")
        raise e

def apply_basic_feature_engineering(df):
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

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para predição"""
    try:
        # Verificar se modelo está carregado
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'status': 'error'
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Dados não fornecidos',
                'status': 'error'
            }), 400
        
        logger.info(f"Recebida requisição de predição: {data}")
        
        # Preprocessar dados
        processed_data = preprocess_input(data)
        
        # Fazer predição
        prediction = model.model.predict(processed_data)[0]
        probability = model.model.predict_proba(processed_data)[0][1]
        
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
        
        # Resposta
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': confidence,
            'recommendation': recommendation,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Predição realizada: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint para predição em lote"""
    try:
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        
        if not data or 'candidates' not in data:
            return jsonify({
                'error': 'Dados de candidatos não fornecidos',
                'status': 'error'
            }), 400
        
        candidates = data['candidates']
        
        if not isinstance(candidates, list):
            return jsonify({
                'error': 'Candidatos devem ser fornecidos como lista',
                'status': 'error'
            }), 400
        
        # Preprocessar dados
        processed_data = preprocess_input(candidates)
        
        # Fazer predições
        predictions = model.model.predict(processed_data)
        probabilities = model.model.predict_proba(processed_data)[:, 1]
        
        # Preparar resposta
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = 'High' if prob > 0.8 else 'Medium' if prob > 0.5 else 'Low'
            recommendation = 'RECOMENDADO' if pred == 1 else 'NÃO RECOMENDADO'
            
            results.append({
                'candidate_index': i,
                'prediction': int(pred),
                'probability': float(prob),
                'confidence': confidence,
                'recommendation': recommendation
            })
        
        response = {
            'results': results,
            'total_candidates': len(candidates),
            'recommended_count': sum(1 for r in results if r['prediction'] == 1),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Predição em lote realizada: {len(candidates)} candidatos")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na predição em lote: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint para informações do modelo"""
    try:
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'status': 'error'
            }), 500
        
        # Carregar metadados
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        response = {
            'model_type': metadata.get('model_type', 'XGBoost'),
            'training_date': metadata.get('training_date'),
            'feature_count': len(model.feature_columns),
            'features': model.feature_columns[:10],  # Primeiras 10 features
            'categorical_features': len(model.categorical_columns),
            'is_trained': metadata.get('is_trained', False),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Endpoint para importância das features"""
    try:
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado',
                'status': 'error'
            }), 500
        
        # Obter importância das features
        importance = model.model.feature_importances_
        feature_names = model.feature_columns
        
        # Criar lista de features com importância
        feature_importance_list = []
        for name, imp in zip(feature_names, importance):
            feature_importance_list.append({
                'feature': name,
                'importance': float(imp)
            })
        
        # Ordenar por importância
        feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        response = {
            'feature_importance': feature_importance_list[:20],  # Top 20
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro ao obter importância das features: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handler para rotas não encontradas"""
    return jsonify({
        'error': 'Endpoint não encontrado',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos"""
    return jsonify({
        'error': 'Erro interno do servidor',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Carregar modelo na inicialização
    logger.info("Iniciando aplicação...")
    
    if load_model():
        logger.info("Aplicação iniciada com sucesso!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Falha ao carregar modelo. Aplicação não iniciada.")
