#!/usr/bin/env python3
"""
Script para validar se o Docker funcionará corretamente em Linux
"""

import os
import sys

def check_dockerfile():
    """Verifica se o Dockerfile está correto para Linux"""
    print("=== VALIDAÇÃO DO DOCKERFILE PARA LINUX ===")
    
    issues = []
    
    # Verificar se Dockerfile existe
    if not os.path.exists('Dockerfile'):
        issues.append("❌ Dockerfile não encontrado")
        return issues
    
    with open('Dockerfile', 'r') as f:
        content = f.read()
    
    # Verificações
    checks = [
        ("FROM python:", "✅ Base image Python definida"),
        ("WORKDIR /app", "✅ Diretório de trabalho definido"),
        ("ENV PYTHONDONTWRITEBYTECODE=1", "✅ Variável de ambiente Python configurada"),
        ("ENV PYTHONUNBUFFERED=1", "✅ Buffer Python desabilitado"),
        ("gcc", "✅ Compilador GCC instalado"),
        ("curl", "✅ Curl instalado para health check"),
        ("EXPOSE 8000", "✅ Porta 8000 exposta"),
        ("uvicorn", "✅ Comando uvicorn configurado"),
        ("0.0.0.0", "✅ Host configurado para aceitar conexões externas")
    ]
    
    for check, message in checks:
        if check in content:
            print(message)
        else:
            issues.append(f"❌ {check} não encontrado")
    
    return issues

def check_requirements():
    """Verifica se requirements.txt está adequado"""
    print("\n=== VALIDAÇÃO DO REQUIREMENTS.TXT ===")
    
    issues = []
    
    if not os.path.exists('requirements.txt'):
        issues.append("❌ requirements.txt não encontrado")
        return issues
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'joblib',
        'pydantic'
    ]
    
    for package in required_packages:
        if package in content:
            print(f"✅ {package} encontrado")
        else:
            issues.append(f"❌ {package} não encontrado")
    
    return issues

def check_dockerignore():
    """Verifica se .dockerignore existe e está adequado"""
    print("\n=== VALIDAÇÃO DO .DOCKERIGNORE ===")
    
    issues = []
    
    if not os.path.exists('.dockerignore'):
        issues.append("❌ .dockerignore não encontrado")
        return issues
    
    with open('.dockerignore', 'r') as f:
        content = f.read()
    
    important_ignores = [
        '__pycache__',
        '*.pyc',
        '.git',
        '.DS_Store',
        'venv/',
        '*.log'
    ]
    
    for ignore in important_ignores:
        if ignore in content:
            print(f"✅ {ignore} será ignorado")
        else:
            print(f"⚠️  {ignore} não está no .dockerignore")
    
    return issues

def check_application_files():
    """Verifica se os arquivos da aplicação existem"""
    print("\n=== VALIDAÇÃO DOS ARQUIVOS DA APLICAÇÃO ===")
    
    issues = []
    
    required_files = [
        'fastapi_app.py',
        'main.py',
        # 'model_pipeline.py',  # Removido (obsoleto)
        'mlflow_config.py',
        'models/',
        'dataset_preparado.csv'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} encontrado")
        else:
            issues.append(f"❌ {file_path} não encontrado")
    
    return issues

def main():
    """Função principal"""
    print("🐧 VALIDAÇÃO DO DOCKER PARA LINUX 🐧")
    print("=" * 50)
    
    all_issues = []
    
    # Executar verificações
    all_issues.extend(check_dockerfile())
    all_issues.extend(check_requirements())
    all_issues.extend(check_dockerignore())
    all_issues.extend(check_application_files())
    
    # Resultado final
    print("\n" + "=" * 50)
    print("📋 RESULTADO DA VALIDAÇÃO")
    print("=" * 50)
    
    if not all_issues:
        print("✅ DOCKER FUNCIONARÁ PERFEITAMENTE EM LINUX!")
        print("\nComandos para usar no Linux:")
        print("1. docker build -t decision-recruitment-ai .")
        print("2. docker run -d -p 8000:8000 --name decision-api decision-recruitment-ai")
        print("3. docker ps  # verificar se está rodando")
        print("4. curl http://localhost:8000/health  # testar API")
        return True
    else:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nCorrija os problemas acima antes de usar no Linux.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
