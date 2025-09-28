#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Deploy para o Sistema de Recrutamento
Decision - Sistema de Match Candidato-Vaga
"""

import os
import subprocess
import sys
import time
import requests
import json
from datetime import datetime

class DeploymentManager:
    """Gerenciador de deploy do sistema"""
    
    def __init__(self):
        self.project_name = "decision-recruitment-ai"
        self.docker_image = f"{self.project_name}:latest"
        self.container_name = "decision-recruitment-api"
        self.port = 8000
        
    def check_requirements(self):
        """Verifica se todos os requisitos estão instalados"""
        print("🔍 Verificando requisitos...")
        
        # Verificar Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker instalado")
            else:
                print("❌ Docker não encontrado")
                return False
        except FileNotFoundError:
            print("❌ Docker não encontrado")
            return False
        
        # Verificar se modelo existe
        if not os.path.exists('models/xgboost_model.pkl'):
            print("❌ Modelo não encontrado. Execute model_pipeline.py primeiro.")
            return False
        else:
            print("✅ Modelo encontrado")
        
        # Verificar se requirements.txt existe
        if not os.path.exists('requirements.txt'):
            print("❌ requirements.txt não encontrado")
            return False
        else:
            print("✅ requirements.txt encontrado")
        
        return True
    
    def build_docker_image(self):
        """Constrói imagem Docker"""
        print(f"\n🔨 Construindo imagem Docker: {self.docker_image}")
        
        try:
            cmd = ['docker', 'build', '-t', self.docker_image, '.']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Imagem Docker construída com sucesso")
                return True
            else:
                print(f"❌ Erro ao construir imagem: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            return False
    
    def stop_existing_container(self):
        """Para container existente se estiver rodando"""
        print(f"\n🛑 Parando container existente: {self.container_name}")
        
        try:
            # Parar container se estiver rodando
            subprocess.run(['docker', 'stop', self.container_name], 
                         capture_output=True)
            
            # Remover container
            subprocess.run(['docker', 'rm', self.container_name], 
                         capture_output=True)
            
            print("✅ Container anterior removido")
            
        except Exception as e:
            print(f"⚠️  Aviso: {str(e)}")
    
    def run_container(self):
        """Executa container Docker"""
        print(f"\n🚀 Iniciando container: {self.container_name}")
        
        try:
            cmd = [
                'docker', 'run', '-d',
                '--name', self.container_name,
                '-p', f'{self.port}:8000',
                '--restart', 'unless-stopped',
                self.docker_image
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Container iniciado com sucesso")
                print(f"   Porta: {self.port}")
                print(f"   URL: http://localhost:{self.port}")
                return True
            else:
                print(f"❌ Erro ao iniciar container: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            return False
    
    def wait_for_api(self, timeout=60):
        """Aguarda API estar pronta"""
        print(f"\n⏳ Aguardando API estar pronta (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", 
                                      timeout=5)
                if response.status_code == 200:
                    print("✅ API está pronta!")
                    return True
            except:
                pass
            
            time.sleep(2)
            print(".", end="", flush=True)
        
        print(f"\n❌ Timeout: API não ficou pronta em {timeout}s")
        return False
    
    def test_deployment(self):
        """Testa o deployment"""
        print(f"\n🧪 Testando deployment...")
        
        try:
            # Teste de health check
            response = requests.get(f"http://localhost:{self.port}/health")
            if response.status_code == 200:
                print("✅ Health check OK")
            else:
                print("❌ Health check falhou")
                return False
            
            # Teste de predição
            test_data = {
                "candidate": {
                    "nome": "Teste Deploy",
                    "nivel_profissional_candidato": "Sênior",
                    "nivel_ingles_candidato": "Avançado",
                    "nivel_espanhol_candidato": "Intermediário",
                    "cv_text": "Python, AWS, Docker",
                    "pcd": "Não",
                    "remuneracao": 8000.0,
                    "estado": "São Paulo"
                },
                "job": {
                    "titulo_vaga": "Desenvolvedor Python",
                    "nivel_profissional_vaga": "Sênior",
                    "nivel_ingles_vaga": "Avançado",
                    "nivel_espanhol_vaga": "Básico",
                    "vaga_sap": "Não",
                    "competencia_tecnicas": "Python, Django, AWS",
                    "cliente": "TechCorp",
                    "tipo_contratacao": "CLT Full"
                }
            }
            
            response = requests.post(f"http://localhost:{self.port}/predict",
                                   json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Predição OK: {result['recommendation']}")
                return True
            else:
                print(f"❌ Predição falhou: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro no teste: {str(e)}")
            return False
    
    def show_logs(self):
        """Mostra logs do container"""
        print(f"\n📋 Logs do container:")
        try:
            result = subprocess.run(['docker', 'logs', '--tail', '20', self.container_name],
                                  capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Erro ao obter logs: {str(e)}")
    
    def show_status(self):
        """Mostra status do deployment"""
        print(f"\n📊 Status do deployment:")
        try:
            result = subprocess.run(['docker', 'ps', '--filter', f'name={self.container_name}'],
                                  capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Erro ao obter status: {str(e)}")
    
    def deploy(self):
        """Executa deploy completo"""
        print("🚀 Iniciando deploy do Decision Recruitment AI")
        print("=" * 60)
        
        # Verificar requisitos
        if not self.check_requirements():
            print("❌ Deploy cancelado: requisitos não atendidos")
            return False
        
        # Parar container existente
        self.stop_existing_container()
        
        # Construir imagem
        if not self.build_docker_image():
            print("❌ Deploy cancelado: erro ao construir imagem")
            return False
        
        # Executar container
        if not self.run_container():
            print("❌ Deploy cancelado: erro ao executar container")
            return False
        
        # Aguardar API
        if not self.wait_for_api():
            print("❌ Deploy cancelado: API não ficou pronta")
            self.show_logs()
            return False
        
        # Testar deployment
        if not self.test_deployment():
            print("❌ Deploy cancelado: testes falharam")
            self.show_logs()
            return False
        
        # Mostrar status final
        self.show_status()
        
        print("\n" + "=" * 60)
        print("🎉 DEPLOY CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print(f"🌐 API disponível em: http://localhost:{self.port}")
        print(f"📚 Documentação: http://localhost:{self.port}/docs")
        print(f"🔍 Health check: http://localhost:{self.port}/health")
        print(f"📊 Status: docker ps --filter name={self.container_name}")
        print(f"📋 Logs: docker logs {self.container_name}")
        
        return True

def main():
    """Função principal"""
    deployer = DeploymentManager()
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deploy realizado com sucesso!")
        print("💡 Use 'python test_api.py' para testar a API")
        print("💡 Use 'python monitoring.py' para monitorar o sistema")
    else:
        print("\n❌ Deploy falhou!")
        sys.exit(1)

if __name__ == "__main__":
    main()
