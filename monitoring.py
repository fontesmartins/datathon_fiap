#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Monitoramento e Logs
Decision - Sistema de Match Candidato-Vaga
"""

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any
import requests
from dataclasses import dataclass, asdict
import time
import mlflow
from mlflow_config import get_mlflow_config

@dataclass
class PredictionLog:
"""Estrutura para log de predições"""
timestamp: str
candidate_name: str
job_title: str
prediction: int
probability: float
confidence: str
recommendation: str
response_time_ms: float
model_version: str
features_used: int

@dataclass
class ModelMetrics:
"""Estrutura para métricas do modelo"""
timestamp: str
total_predictions: int
successful_predictions: int
failed_predictions: int
avg_response_time_ms: float
avg_probability: float
recommendation_rate: float
high_confidence_rate: float

class ModelMonitor:
"""Monitor de modelo para detecção de drift e métricas"""

def __init__(self, log_file: str = "logs/predictions.log", 
metrics_file: str = "logs/metrics.json"):
self.log_file = log_file
self.metrics_file = metrics_file
self.setup_logging()
self.setup_directories()

def setup_logging(self):
"""Configura sistema de logging"""
# Criar diretório de logs se não existir
os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

# Configurar logger
self.logger = logging.getLogger('model_monitor')
self.logger.setLevel(logging.INFO)

# Handler para arquivo
file_handler = logging.FileHandler(self.log_file)
file_handler.setLevel(logging.INFO)

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato
formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

self.logger.addHandler(file_handler)
self.logger.addHandler(console_handler)

def setup_directories(self):
"""Cria diretórios necessários"""
os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

def log_prediction(self, prediction_data: Dict[str, Any], 
response_time_ms: float, model_version: str = "1.0.0"):
"""Registra uma predição no log"""
try:
log_entry = PredictionLog(
timestamp=datetime.now().isoformat(),
candidate_name=prediction_data.get('candidate_name', 'Unknown'),
job_title=prediction_data.get('job_title', 'Unknown'),
prediction=prediction_data.get('prediction', 0),
probability=prediction_data.get('probability', 0.0),
confidence=prediction_data.get('confidence', 'Low'),
recommendation=prediction_data.get('recommendation', 'NÃO RECOMENDADO'),
response_time_ms=response_time_ms,
model_version=model_version,
features_used=prediction_data.get('features_used', 0)
)

# Log estruturado
self.logger.info(f"PREDICTION: {json.dumps(asdict(log_entry))}")

return True

except Exception as e:
self.logger.error(f"Erro ao registrar predição: {str(e)}")
return False

def calculate_metrics(self, time_window_hours: int = 24) -> ModelMetrics:
"""Calcula métricas do modelo para um período"""
try:
# Ler logs do período
logs = self.read_prediction_logs(time_window_hours)

if not logs:
return ModelMetrics(
timestamp=datetime.now().isoformat(),
total_predictions=0,
successful_predictions=0,
failed_predictions=0,
avg_response_time_ms=0.0,
avg_probability=0.0,
recommendation_rate=0.0,
high_confidence_rate=0.0
)

# Calcular métricas
total_predictions = len(logs)
successful_predictions = len([log for log in logs if log.get('prediction') is not None])
failed_predictions = total_predictions - successful_predictions

avg_response_time = np.mean([log.get('response_time_ms', 0) for log in logs])
avg_probability = np.mean([log.get('probability', 0) for log in logs])

recommendations = [log for log in logs if log.get('recommendation') == 'RECOMENDADO']
recommendation_rate = len(recommendations) / total_predictions if total_predictions > 0 else 0

high_confidence = [log for log in logs if log.get('confidence') == 'High']
high_confidence_rate = len(high_confidence) / total_predictions if total_predictions > 0 else 0

metrics = ModelMetrics(
timestamp=datetime.now().isoformat(),
total_predictions=total_predictions,
successful_predictions=successful_predictions,
failed_predictions=failed_predictions,
avg_response_time_ms=avg_response_time,
avg_probability=avg_probability,
recommendation_rate=recommendation_rate,
high_confidence_rate=high_confidence_rate
)

return metrics

except Exception as e:
self.logger.error(f"Erro ao calcular métricas: {str(e)}")
return None

def read_prediction_logs(self, time_window_hours: int = 24) -> List[Dict]:
"""Lê logs de predições de um período"""
try:
logs = []
cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

if not os.path.exists(self.log_file):
return logs

with open(self.log_file, 'r') as f:
for line in f:
if 'PREDICTION:' in line:
try:
# Extrair JSON do log
json_start = line.find('{')
if json_start != -1:
json_str = line[json_start:]
log_data = json.loads(json_str)

# Verificar se está no período
log_time = datetime.fromisoformat(log_data['timestamp'])
if log_time >= cutoff_time:
logs.append(log_data)
except json.JSONDecodeError:
continue

return logs

except Exception as e:
self.logger.error(f"Erro ao ler logs: {str(e)}")
return []

def detect_drift(self, baseline_metrics: ModelMetrics, 
current_metrics: ModelMetrics, 
threshold: float = 0.1) -> Dict[str, Any]:
"""Detecta drift no modelo comparando métricas"""
drift_detected = False
drift_details = {}

try:
# Comparar taxa de recomendação
rec_diff = abs(current_metrics.recommendation_rate - baseline_metrics.recommendation_rate)
if rec_diff > threshold:
drift_detected = True
drift_details['recommendation_rate_drift'] = {
'baseline': baseline_metrics.recommendation_rate,
'current': current_metrics.recommendation_rate,
'difference': rec_diff
}

# Comparar probabilidade média
prob_diff = abs(current_metrics.avg_probability - baseline_metrics.avg_probability)
if prob_diff > threshold:
drift_detected = True
drift_details['probability_drift'] = {
'baseline': baseline_metrics.avg_probability,
'current': current_metrics.avg_probability,
'difference': prob_diff
}

# Comparar taxa de alta confiança
conf_diff = abs(current_metrics.high_confidence_rate - baseline_metrics.high_confidence_rate)
if conf_diff > threshold:
drift_detected = True
drift_details['confidence_drift'] = {
'baseline': baseline_metrics.high_confidence_rate,
'current': current_metrics.high_confidence_rate,
'difference': conf_diff
}

return {
'drift_detected': drift_detected,
'timestamp': datetime.now().isoformat(),
'details': drift_details,
'threshold_used': threshold
}

except Exception as e:
self.logger.error(f"Erro na detecção de drift: {str(e)}")
return {
'drift_detected': False,
'error': str(e),
'timestamp': datetime.now().isoformat()
}

def save_metrics(self, metrics: ModelMetrics):
"""Salva métricas em arquivo"""
try:
metrics_data = asdict(metrics)

# Carregar métricas existentes
if os.path.exists(self.metrics_file):
with open(self.metrics_file, 'r') as f:
all_metrics = json.load(f)
else:
all_metrics = []

# Adicionar nova métrica
all_metrics.append(metrics_data)

# Manter apenas últimas 100 entradas
if len(all_metrics) > 100:
all_metrics = all_metrics[-100:]

# Salvar
with open(self.metrics_file, 'w') as f:
json.dump(all_metrics, f, indent=2)

self.logger.info(f"Métricas salvas: {metrics.total_predictions} predições")

except Exception as e:
self.logger.error(f"Erro ao salvar métricas: {str(e)}")

def generate_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
"""Gera relatório de monitoramento"""
try:
metrics = self.calculate_metrics(time_window_hours)

report = {
'report_timestamp': datetime.now().isoformat(),
'time_window_hours': time_window_hours,
'metrics': asdict(metrics) if metrics else None,
'system_status': 'healthy' if metrics and metrics.failed_predictions == 0 else 'warning',
'recommendations': []
}

# Adicionar recomendações baseadas nas métricas
if metrics:
if metrics.avg_response_time_ms > 1000:
report['recommendations'].append("Tempo de resposta alto - considerar otimização")

if metrics.recommendation_rate < 0.1:
report['recommendations'].append("Taxa de recomendação baixa - revisar critérios")

if metrics.high_confidence_rate < 0.3:
report['recommendations'].append("Taxa de alta confiança baixa - revisar features")

return report

except Exception as e:
self.logger.error(f"Erro ao gerar relatório: {str(e)}")
return {
'error': str(e),
'timestamp': datetime.now().isoformat()
}

class MLflowMonitor:
"""Monitor para MLflow experiments e models"""

def __init__(self):
self.mlflow_config = get_mlflow_config()
self.client = mlflow.tracking.MlflowClient()

def get_experiment_metrics(self) -> Dict[str, Any]:
"""Obtém métricas dos experimentos MLflow"""
try:
experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
if experiment is None:
return {"error": "Experimento não encontrado"}

# Buscar todos os runs do experimento
runs = self.client.search_runs(
experiment_ids=[experiment.experiment_id],
max_results=100
)

if not runs:
return {"error": "Nenhum run encontrado"}

# Calcular métricas agregadas
auc_scores = []
cv_auc_scores = []
run_times = []

for run in runs:
if run.data.metrics:
if 'auc_score' in run.data.metrics:
auc_scores.append(run.data.metrics['auc_score'])
if 'cv_auc_mean' in run.data.metrics:
cv_auc_scores.append(run.data.metrics['cv_auc_mean'])

if run.info.end_time and run.info.start_time:
run_times.append((run.info.end_time - run.info.start_time) / 1000) # em segundos

metrics = {
'total_runs': len(runs),
'successful_runs': len([r for r in runs if r.info.status == 'FINISHED']),
'failed_runs': len([r for r in runs if r.info.status == 'FAILED']),
'avg_auc_score': np.mean(auc_scores) if auc_scores else 0,
'max_auc_score': np.max(auc_scores) if auc_scores else 0,
'avg_cv_auc_score': np.mean(cv_auc_scores) if cv_auc_scores else 0,
'max_cv_auc_score': np.max(cv_auc_scores) if cv_auc_scores else 0,
'avg_run_time_seconds': np.mean(run_times) if run_times else 0,
'experiment_name': experiment.name,
'experiment_id': experiment.experiment_id
}

return metrics

except Exception as e:
return {"error": f"Erro ao obter métricas: {str(e)}"}

def get_model_registry_info(self) -> Dict[str, Any]:
"""Obtém informações do Model Registry"""
try:
# Buscar modelos registrados
registered_models = self.client.search_registered_models()

model_info = {
'total_models': len(registered_models),
'models': []
}

for model in registered_models:
model_data = {
'name': model.name,
'latest_version': model.latest_versions[0].version if model.latest_versions else None,
'stages': [v.current_stage for v in model.latest_versions],
'creation_timestamp': model.creation_timestamp,
'last_updated_timestamp': model.last_updated_timestamp
}
model_info['models'].append(model_data)

return model_info

except Exception as e:
return {"error": f"Erro ao obter informações do registry: {str(e)}"}

def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
"""Obtém runs recentes"""
try:
experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
if experiment is None:
return []

runs = self.client.search_runs(
experiment_ids=[experiment.experiment_id],
max_results=limit,
order_by=["start_time DESC"]
)

recent_runs = []
for run in runs:
run_data = {
'run_id': run.info.run_id,
'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
'start_time': run.info.start_time,
'end_time': run.info.end_time,
'status': run.info.status,
'metrics': dict(run.data.metrics),
'params': dict(run.data.params),
'tags': dict(run.data.tags)
}
recent_runs.append(run_data)

return recent_runs

except Exception as e:
return []

class APIMonitor:
"""Monitor para API endpoints"""

def __init__(self, base_url: str = "http://localhost:8000"):
self.base_url = base_url
self.monitor = ModelMonitor()
self.mlflow_monitor = MLflowMonitor()

def check_api_health(self) -> Dict[str, Any]:
"""Verifica saúde da API"""
try:
start_time = time.time()
response = requests.get(f"{self.base_url}/health", timeout=10)
response_time = (time.time() - start_time) * 1000

return {
'status': 'healthy' if response.status_code == 200 else 'unhealthy',
'status_code': response.status_code,
'response_time_ms': response_time,
'timestamp': datetime.now().isoformat()
}

except Exception as e:
return {
'status': 'unhealthy',
'error': str(e),
'timestamp': datetime.now().isoformat()
}

def test_prediction_endpoint(self) -> Dict[str, Any]:
"""Testa endpoint de predição"""
try:
test_data = {
"candidate": {
"nome": "Teste Monitor",
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

start_time = time.time()
response = requests.post(
f"{self.base_url}/predict",
json=test_data,
timeout=30
)
response_time = (time.time() - start_time) * 1000

if response.status_code == 200:
result = response.json()

# Log da predição
self.monitor.log_prediction(
{
'candidate_name': test_data['candidate']['nome'],
'job_title': test_data['job']['titulo_vaga'],
'prediction': result['prediction'],
'probability': result['probability'],
'confidence': result['confidence'],
'recommendation': result['recommendation'],
'features_used': 46 # Número de features do modelo
},
response_time
)

return {
'status': 'success',
'response_time_ms': response_time,
'prediction': result,
'timestamp': datetime.now().isoformat()
}
else:
return {
'status': 'error',
'status_code': response.status_code,
'error': response.text,
'timestamp': datetime.now().isoformat()
}

except Exception as e:
return {
'status': 'error',
'error': str(e),
'timestamp': datetime.now().isoformat()
}

def main():
"""Função principal de monitoramento"""
print(" Iniciando sistema de monitoramento...")

# Inicializar monitor
monitor = ModelMonitor()
api_monitor = APIMonitor()
mlflow_monitor = MLflowMonitor()

# Verificar saúde da API
print(" Verificando saúde da API...")
health = api_monitor.check_api_health()
print(f" Status: {health['status']}")
print(f" Tempo de resposta: {health.get('response_time_ms', 0):.2f}ms")

# Testar endpoint de predição
print("\n Testando endpoint de predição...")
prediction_test = api_monitor.test_prediction_endpoint()
print(f" Status: {prediction_test['status']}")
if prediction_test['status'] == 'success':
print(f" Tempo de resposta: {prediction_test['response_time_ms']:.2f}ms")
print(f" Recomendação: {prediction_test['prediction']['recommendation']}")

# Calcular métricas
print("\n Calculando métricas...")
metrics = monitor.calculate_metrics(24)
if metrics:
print(f" Total de predições (24h): {metrics.total_predictions}")
print(f" Taxa de recomendação: {metrics.recommendation_rate:.2%}")
print(f" Taxa de alta confiança: {metrics.high_confidence_rate:.2%}")
print(f" Tempo médio de resposta: {metrics.avg_response_time_ms:.2f}ms")

# Salvar métricas
monitor.save_metrics(metrics)

# Monitorar MLflow
print("\n Verificando MLflow...")
try:
# Métricas dos experimentos
experiment_metrics = mlflow_monitor.get_experiment_metrics()
if 'error' not in experiment_metrics:
print(f" Total de runs: {experiment_metrics['total_runs']}")
print(f" Runs bem-sucedidos: {experiment_metrics['successful_runs']}")
print(f" Melhor AUC Score: {experiment_metrics['max_auc_score']:.4f}")
print(f" Melhor CV AUC Score: {experiment_metrics['max_cv_auc_score']:.4f}")
else:
print(f" Erro: {experiment_metrics['error']}")

# Informações do Model Registry
registry_info = mlflow_monitor.get_model_registry_info()
if 'error' not in registry_info:
print(f" Modelos registrados: {registry_info['total_models']}")
for model in registry_info['models']:
print(f" - {model['name']}: v{model['latest_version']} ({', '.join(model['stages'])})")
else:
print(f" Erro no registry: {registry_info['error']}")

# Runs recentes
recent_runs = mlflow_monitor.get_recent_runs(5)
if recent_runs:
print(f" Runs recentes: {len(recent_runs)}")
for run in recent_runs[:3]: # Mostrar apenas os 3 mais recentes
print(f" - {run['run_name']}: {run['status']} (AUC: {run['metrics'].get('auc_score', 'N/A')})")

except Exception as e:
print(f" Erro ao monitorar MLflow: {e}")

# Gerar relatório
print("\n Gerando relatório...")
report = monitor.generate_report(24)
print(f" Status do sistema: {report['system_status']}")
if report['recommendations']:
print(" Recomendações:")
for rec in report['recommendations']:
print(f" - {rec}")

print("\n Monitoramento concluído!")
print("\n Para visualizar MLflow UI:")
print(" mlflow ui --backend-store-uri file:./mlruns")
print(" Acesse: http://localhost:5000")

if __name__ == "__main__":
main()
