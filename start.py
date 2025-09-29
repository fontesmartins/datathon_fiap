#!/usr/bin/env python3
"""
Script de inicialização para Railway
Verifica modelos antes de iniciar a API
"""

import os
import sys
import subprocess

def check_and_start():
    """Verifica modelos e inicia a API"""
    print("🚀 Iniciando Decision Recruitment AI...")
    
    # Verificar se estamos no Railway
    railway_env = os.environ.get('RAILWAY_ENVIRONMENT')
    if railway_env:
        print(f"🏗️ Executando no Railway - Environment: {railway_env}")
    else:
        print("💻 Executando localmente")
    
    # Verificar arquivos necessários
    required_files = [
        'models/xgboost_model.pkl',
        'models/label_encoders.pkl',
        'models/scaler.pkl',
        'models/model_metadata.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ ERRO: Arquivos ausentes: {missing_files}")
        print("💡 Execute 'python main.py' para treinar o modelo")
        sys.exit(1)
    
    print("✅ Todos os arquivos de modelo estão presentes")
    
    # Obter porta do Railway
    port = os.environ.get('PORT', '8000')
    print(f"🌐 Iniciando na porta: {port}")
    
    # Iniciar uvicorn
    cmd = [
        'uvicorn', 
        'fastapi_app:app', 
        '--host', '0.0.0.0', 
        '--port', port,
        '--workers', '1',
        '--log-level', 'info'
    ]
    
    print(f"🚀 Executando: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    check_and_start()
