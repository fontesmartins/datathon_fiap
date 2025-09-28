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

print(f"\n Testando: {description}")
print(f" {method} {url}")

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
raise ValueError(f"Método {method} não suportado")

result = {
"endpoint": endpoint,
"method": method,
"description": description,
"status_code": response.status_code,
"success": response.status_code == expected_status,
"response_time": response.elapsed.total_seconds(),
"timestamp": datetime.now().isoformat()
}

try:
result["response_data"] = response.json()
except:
result["response_data"] = response.text

if result["success"]:
print(f" Sucesso ({result['response_time']:.3f}s)")
if isinstance(result["response_data"], dict) and "status" in result["response_data"]:
print(f" Status: {result['response_data']['status']}")
else:
print(f" Erro - Status: {response.status_code}")
print(f" Resposta: {result['response_data']}")

self.results.append(result)
return result

except requests.exceptions.RequestException as e:
error_result = {
"endpoint": endpoint,
"method": method,
"description": description,
"error": str(e),
"success": False,
"timestamp": datetime.now().isoformat()
}
print(f" Erro de conexão: {e}")
self.results.append(error_result)
return error_result

def test_health_check(self):
"""Testa o health check"""
return self.test_endpoint(
"GET", "/health", 
description="Health Check"
)

def test_model_info(self):
"""Testa informações do modelo"""
return self.test_endpoint(
"GET", "/model_info",
description="Informações do Modelo"
)

def test_feature_importance(self):
"""Testa importância das features"""
return self.test_endpoint(
"GET", "/feature_importance",
description="Importância das Features"
)

def test_mlflow_info(self):
"""Testa informações do MLflow"""
return self.test_endpoint(
"GET", "/mlflow_info",
description="Informações do MLflow"
)

def test_prediction(self, candidate_data: Dict, job_data: Dict, description: str):
"""Testa predição individual"""
data = {
"candidate": candidate_data,
"job": job_data
}
return self.test_endpoint(
"POST", "/predict", data,
description=description
)

def test_batch_prediction(self, candidates: List[Dict], job_data: Dict, description: str):
"""Testa predição em lote"""
data = {
"candidates": candidates,
"job": job_data
}
return self.test_endpoint(
"POST", "/predict_batch", data,
description=description
)

def run_all_tests(self):
"""Executa todos os testes"""
print(" Iniciando Testes da API Decision Recruitment AI")
print(f" URL Base: {self.base_url}")
print("=" * 60)

# Testes básicos
self.test_health_check()
self.test_model_info()
self.test_feature_importance()
self.test_mlflow_info()

# Dados de teste
candidate_python = {
"nome": "João Silva",
"nivel_profissional_candidato": "Sênior",
"nivel_ingles_candidato": "Avançado",
"nivel_espanhol_candidato": "Intermediário",
"cv_text": "Desenvolvedor Python com 5 anos de experiência em desenvolvimento web, APIs REST, Django, Flask, PostgreSQL, AWS, Docker, Kubernetes. Certificado AWS Solutions Architect.",
"pcd": "Não",
"remuneracao": 12000,
"estado": "São Paulo"
}

job_python = {
"titulo_vaga": "Desenvolvedor Python Sênior",
"nivel_profissional_vaga": "Sênior",
"nivel_ingles_vaga": "Avançado",
"nivel_espanhol_vaga": "Intermediário",
"vaga_sap": "Não",
"competencia_tecnicas": "Python, Django, Flask, PostgreSQL, AWS, Docker, Kubernetes, APIs REST",
"cliente": "TechCorp",
"tipo_contratacao": "CLT Full"
}

candidate_sap = {
"nome": "Maria Santos",
"nivel_profissional_candidato": "Pleno",
"nivel_ingles_candidato": "Fluente",
"nivel_espanhol_candidato": "Fluente",
"cv_text": "Analista de Sistemas com experiência em SAP, SQL Server, Oracle, desenvolvimento de relatórios, análise de dados, Excel avançado, VBA, Power BI.",
"pcd": "Não",
"remuneracao": 8000,
"estado": "São Paulo"
}

job_sap = {
"titulo_vaga": "Analista SAP Pleno",
"nivel_profissional_vaga": "Pleno",
"nivel_ingles_vaga": "Fluente",
"nivel_espanhol_vaga": "Fluente",
"vaga_sap": "Sim",
"competencia_tecnicas": "SAP, SQL, Oracle, Excel, VBA, Power BI, Análise de Dados",
"cliente": "Morris, Moran and Dodson",
"tipo_contratacao": "CLT Full"
}

# Testes de predição
self.test_prediction(candidate_python, job_python, "Predição - Desenvolvedor Python")
self.test_prediction(candidate_sap, job_sap, "Predição - Analista SAP")

# Teste de predição em lote
candidates_batch = [
{
"nome": "Ana Costa",
"nivel_profissional_candidato": "Júnior",
"nivel_ingles_candidato": "Básico",
"nivel_espanhol_candidato": "Nenhum",
"cv_text": "Desenvolvedora iniciante com conhecimento em Python, HTML, CSS, JavaScript.",
"pcd": "Não",
"remuneracao": 3000,
"estado": "São Paulo"
},
{
"nome": "Carlos Oliveira",
"nivel_profissional_candidato": "Sênior",
"nivel_ingles_candidato": "Fluente",
"nivel_espanhol_candidato": "Avançado",
"cv_text": "Arquiteto de Software com 8 anos de experiência em Java, Spring Boot, Microservices, AWS, Docker, Kubernetes.",
"pcd": "Não",
"remuneracao": 15000,
"estado": "São Paulo"
}
]

job_batch = {
"titulo_vaga": "Desenvolvedor Java Sênior",
"nivel_profissional_vaga": "Sênior",
"nivel_ingles_vaga": "Fluente",
"nivel_espanhol_vaga": "Avançado",
"vaga_sap": "Não",
"competencia_tecnicas": "Java, Spring Boot, Microservices, AWS, Docker, Kubernetes",
"cliente": "TechCorp",
"tipo_contratacao": "CLT Full"
}

self.test_batch_prediction(candidates_batch, job_batch, "Predição em Lote")

# Resumo dos resultados
self.print_summary()

def print_summary(self):
"""Imprime resumo dos testes"""
print("\n" + "=" * 60)
print(" RESUMO DOS TESTES")
print("=" * 60)

total_tests = len(self.results)
successful_tests = sum(1 for r in self.results if r.get("success", False))
failed_tests = total_tests - successful_tests

print(f" Testes bem-sucedidos: {successful_tests}")
print(f" Testes falharam: {failed_tests}")
print(f" Taxa de sucesso: {(successful_tests/total_tests)*100:.1f}%")

if failed_tests > 0:
print("\n Testes que falharam:")
for result in self.results:
if not result.get("success", False):
print(f" - {result['description']}: {result.get('error', 'Status code incorreto')}")

# Estatísticas de tempo
response_times = [r.get("response_time", 0) for r in self.results if "response_time" in r]
if response_times:
avg_time = sum(response_times) / len(response_times)
max_time = max(response_times)
min_time = min(response_times)
print(f"\n⏱ Tempo de resposta:")
print(f" - Média: {avg_time:.3f}s")
print(f" - Máximo: {max_time:.3f}s")
print(f" - Mínimo: {min_time:.3f}s")

print(f"\n Documentação disponível em: {self.base_url}/docs")
print(f" ReDoc disponível em: {self.base_url}/redoc")

# Salvar resultados em arquivo
with open("test_results.json", "w") as f:
json.dump(self.results, f, indent=2, ensure_ascii=False)
print(f" Resultados salvos em: test_results.json")

def main():
import sys

# URL base da API
base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

# Executar testes
tester = APITester(base_url)
tester.run_all_tests()

if __name__ == "__main__":
main()
