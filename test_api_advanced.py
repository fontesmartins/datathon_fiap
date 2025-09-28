#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste Avançado da API Decision Recruitment AI
Testa todos os endpoints com validação de respostas
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = []

    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, 
                     expected_status: int = 200, description: str = "") -> Dict[str, Any]:
        """Testa um endpoint específico"""
        url = f"{self.base_url}{endpoint}"
        
        print(f"\n🔍 Testando: {description}")
        print(f"   {method} {url}")
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(
                    url, 
                    json=data, 
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")
            
            # Verificar status code
            status_ok = response.status_code == expected_status
            status_icon = "✅" if status_ok else "❌"
            
            print(f"   Status: {status_icon} {response.status_code}")
            
            # Tentar parsear JSON
            try:
                response_data = response.json()
                print(f"   Resposta: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"   Resposta (texto): {response.text[:200]}...")
                response_data = None
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": status_ok,
                "response_data": response_data,
                "timestamp": datetime.now().isoformat()
            }
            
            if status_ok:
                print(f"   ✅ Teste passou!")
            else:
                print(f"   ❌ Teste falhou! Esperado: {expected_status}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erro de conexão: {e}")
            return {
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status_code": None,
                "expected_status": expected_status,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"   ❌ Erro inesperado: {e}")
            return {
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status_code": None,
                "expected_status": expected_status,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def test_health_endpoint(self):
        """Testa endpoint de health check"""
        return self.test_endpoint(
            "GET", 
            "/health", 
            description="Health Check"
        )

    def test_model_info_endpoint(self):
        """Testa endpoint de informações do modelo"""
        return self.test_endpoint(
            "GET", 
            "/model_info", 
            description="Informações do Modelo"
        )

    def test_feature_importance_endpoint(self):
        """Testa endpoint de importância das features"""
        return self.test_endpoint(
            "GET", 
            "/feature_importance", 
            description="Importância das Features"
        )

    def test_mlflow_info_endpoint(self):
        """Testa endpoint de informações MLflow"""
        return self.test_endpoint(
            "GET", 
            "/mlflow_info", 
            description="Informações MLflow"
        )

    def test_predict_endpoint(self):
        """Testa endpoint de predição"""
        test_data = {
            "candidate": {
                "nome": "João Silva",
                "nivel_profissional_candidato": "Sênior",
                "nivel_ingles_candidato": "Avançado",
                "nivel_espanhol_candidato": "Intermediário",
                "cv_text": "Desenvolvedor Python com 5 anos de experiência em Django, Flask, AWS, Docker. Especialista em machine learning e análise de dados.",
                "pcd": "Não",
                "remuneracao": 8000.0,
                "estado": "São Paulo"
            },
            "job": {
                "titulo_vaga": "Desenvolvedor Python Sênior",
                "nivel_profissional_vaga": "Sênior",
                "nivel_ingles_vaga": "Avançado",
                "nivel_espanhol_vaga": "Básico",
                "vaga_sap": "Não",
                "competencia_tecnicas": "Python, Django, Flask, AWS, Docker",
                "tipo_contratacao": "CLT Full"
            }
        }
        
        return self.test_endpoint(
            "POST", 
            "/predict", 
            data=test_data,
            description="Predição Individual"
        )

    def test_batch_predict_endpoint(self):
        """Testa endpoint de predição em lote"""
        test_data = {
            "candidates": [
                {
                    "nome": "Maria Santos",
                    "nivel_profissional_candidato": "Pleno",
                    "nivel_ingles_candidato": "Intermediário",
                    "nivel_espanhol_candidato": "Básico",
                    "cv_text": "Desenvolvedora Java com 3 anos de experiência. Conhecimento em Spring Boot, MySQL, Git.",
                    "pcd": "Não",
                    "remuneracao": 6000.0,
                    "estado": "Rio de Janeiro"
                },
                {
                    "nome": "Pedro Costa",
                    "nivel_profissional_candidato": "Júnior",
                    "nivel_ingles_candidato": "Básico",
                    "nivel_espanhol_candidato": "Nenhum",
                    "cv_text": "Estudante de Ciência da Computação. Conhecimento básico em Python, HTML, CSS.",
                    "pcd": "Não",
                    "remuneracao": 3000.0,
                    "estado": "Minas Gerais"
                }
            ],
            "job": {
                "titulo_vaga": "Desenvolvedor Java Pleno",
                "nivel_profissional_vaga": "Pleno",
                "nivel_ingles_vaga": "Intermediário",
                "nivel_espanhol_vaga": "Nenhum",
                "vaga_sap": "Não",
                "competencia_tecnicas": "Java, Spring Boot, MySQL, Git",
                "tipo_contratacao": "CLT Full"
            }
        }
        
        return self.test_endpoint(
            "POST", 
            "/predict_batch", 
            data=test_data,
            description="Predição em Lote"
        )

    def run_all_tests(self):
        """Executa todos os testes"""
        print("🚀 INICIANDO TESTES AVANÇADOS DA API")
        print("=" * 60)
        print(f"🌐 URL Base: {self.base_url}")
        print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Lista de testes
        tests = [
            self.test_health_endpoint,
            self.test_model_info_endpoint,
            self.test_feature_importance_endpoint,
            self.test_mlflow_info_endpoint,
            self.test_predict_endpoint,
            self.test_batch_predict_endpoint
        ]
        
        # Executar testes
        for test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
            except Exception as e:
                print(f"❌ Erro ao executar teste {test_func.__name__}: {e}")
                self.results.append({
                    "test_function": test_func.__name__,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Pequena pausa entre testes
            time.sleep(0.5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Relatório final
        self.print_summary(duration)
        
        return self.results

    def print_summary(self, duration: float):
        """Imprime resumo dos testes"""
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get("success", False))
        failed_tests = total_tests - successful_tests
        
        print(f"⏱️  Duração total: {duration:.2f} segundos")
        print(f"📈 Total de testes: {total_tests}")
        print(f"✅ Testes bem-sucedidos: {successful_tests}")
        print(f"❌ Testes falharam: {failed_tests}")
        print(f"📊 Taxa de sucesso: {(successful_tests/total_tests*100):.1f}%")
        
        # Detalhes dos testes
        print(f"\n📋 DETALHES DOS TESTES:")
        print("-" * 40)
        
        for result in self.results:
            status_icon = "✅" if result.get("success", False) else "❌"
            description = result.get("description", "Teste sem descrição")
            status_code = result.get("status_code", "N/A")
            
            print(f"{status_icon} {description} - Status: {status_code}")
            
            if not result.get("success", False) and "error" in result:
                print(f"   💥 Erro: {result['error']}")
        
        # Salvar resultados
        self.save_results()

    def save_results(self):
        """Salva resultados em arquivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_summary": {
                        "timestamp": datetime.now().isoformat(),
                        "base_url": self.base_url,
                        "total_tests": len(self.results),
                        "successful_tests": sum(1 for r in self.results if r.get("success", False)),
                        "failed_tests": len(self.results) - sum(1 for r in self.results if r.get("success", False))
                    },
                    "test_results": self.results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Resultados salvos em: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    import sys
    
    # URL base (padrão ou argumento)
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print("Decision Recruitment AI - Teste Avançado da API")
    print("=" * 50)
    print(f"🌐 Testando API em: {base_url}")
    
    # Criar tester e executar testes
    tester = APITester(base_url)
    results = tester.run_all_tests()
    
    # Verificar se todos os testes passaram
    all_passed = all(r.get("success", False) for r in results)
    
    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        sys.exit(0)
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM!")
        sys.exit(1)

if __name__ == "__main__":
    main()
