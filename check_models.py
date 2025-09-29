#!/usr/bin/env python3
"""
Script para verificar se todos os modelos estão presentes antes do deploy
"""

import os
import sys
import joblib

def check_models():
    """Verifica se todos os arquivos de modelo estão presentes"""
    required_files = [
        'models/xgboost_model.pkl',
        'models/label_encoders.pkl', 
        'models/scaler.pkl',
        'models/model_metadata.json',
        'dataset_preparado.csv'
    ]
    
    missing_files = []
    
    print("🔍 Verificando arquivos necessários...")
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} - {size:,} bytes")
        else:
            print(f"❌ {file_path} - AUSENTE")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ ERRO: {len(missing_files)} arquivos ausentes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Execute 'python main.py' para treinar o modelo")
        return False
    
    print("\n✅ Todos os arquivos estão presentes!")
    
    # Tentar carregar os modelos
    print("\n🧪 Testando carregamento dos modelos...")
    
    try:
        model = joblib.load('models/xgboost_model.pkl')
        print(f"✅ Modelo XGBoost carregado - {model.n_features_in_} features")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    try:
        encoders = joblib.load('models/label_encoders.pkl')
        print(f"✅ Label encoders carregados - {len(encoders)} encoders")
    except Exception as e:
        print(f"❌ Erro ao carregar encoders: {e}")
        return False
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        print(f"✅ Scaler carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar scaler: {e}")
        return False
    
    print("\n🚀 PRONTO PARA DEPLOY!")
    return True

if __name__ == "__main__":
    success = check_models()
    sys.exit(0 if success else 1)
