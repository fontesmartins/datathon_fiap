# Guia de Testes da API Decision Recruitment AI

Este diretório contém todos os arquivos necessários para testar a API Decision Recruitment AI, tanto localmente quanto na nuvem.

## Arquivos de Teste

### Scripts de Teste
- **`test_api.sh`** - Script bash para testes básicos
- **`test_api_advanced.py`** - Script Python para testes avançados com validação
- **`TESTE_API_RESUMO.md`** - Resumo dos testes realizados

### Dados de Exemplo
- **`test_prediction.json`** - Exemplo de candidato desenvolvedor Python
- **`test_prediction_positive.json`** - Exemplo de candidato analista SAP
- **`test_batch_prediction.json`** - Exemplo de predição em lote

## Como Executar os Testes

### 1. Testes Básicos (Bash)

```bash
# Teste local
./test_api.sh

# Teste na nuvem (substitua pela sua URL)
./test_api.sh https://sua-api.herokuapp.com
```

### 2. Testes Avançados (Python)

```bash
# Teste local
python test_api_advanced.py

# Teste na nuvem
python test_api_advanced.py https://sua-api.herokuapp.com
```

### 3. Testes Manuais com cURL

```bash
# Health check
curl -X GET "https://sua-api.herokuapp.com/health"

# Informações do modelo
curl -X GET "https://sua-api.herokuapp.com/model_info"

# Predição individual
curl -X POST "https://sua-api.herokuapp.com/predict" \
-H "Content-Type: application/json" \
-d @test_prediction.json

# Predição em lote
curl -X POST "https://sua-api.herokuapp.com/predict_batch" \
-H "Content-Type: application/json" \
-d @test_batch_prediction.json
```

## Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações básicas da API |
| `/health` | GET | Status de saúde da API |
| `/model_info` | GET | Informações do modelo ML |
| `/feature_importance` | GET | Importância das features |
| `/mlflow_info` | GET | Informações do MLflow |
| `/predict` | POST | Predição individual |
| `/predict_batch` | POST | Predição em lote |
| `/docs` | GET | Documentação Swagger UI |
| `/redoc` | GET | Documentação ReDoc |

## Estrutura dos Dados

### Candidato
```json
{
"nome": "string",
"nivel_profissional_candidato": "Júnior|Pleno|Sênior|Especialista|Líder",
"nivel_ingles_candidato": "Nenhum|Básico|Intermediário|Avançado|Fluente",
"nivel_espanhol_candidato": "Nenhum|Básico|Intermediário|Avançado|Fluente",
"cv_text": "string",
"pcd": "Sim|Não",
"remuneracao": number,
"estado": "string"
}
```

### Vaga
```json
{
"titulo_vaga": "string",
"nivel_profissional_vaga": "Júnior|Pleno|Sênior|Especialista|Líder",
"nivel_ingles_vaga": "Nenhum|Básico|Intermediário|Avançado|Fluente",
"nivel_espanhol_vaga": "Nenhum|Básico|Intermediário|Avançado|Fluente",
"vaga_sap": "Sim|Não",
"competencia_tecnicas": "string",
"cliente": "string",
"tipo_contratacao": "string"
}
```

## Resposta da Predição

```json
{
"prediction": 0|1,
"probability": 0.0-1.0,
"confidence": "Low|Medium|High",
"recommendation": "RECOMENDADO|NÃO RECOMENDADO",
"explanation": "string",
"timestamp": "ISO datetime",
"status": "success"
}
```

## Exemplos de Uso

### 1. Desenvolvedor Python Sênior
- **Nível:** Sênior
- **Idiomas:** Inglês Avançado, Espanhol Intermediário
- **CV:** Experiência em Python, Django, Flask, AWS, Docker
- **Resultado esperado:** Depende da compatibilidade com a vaga

### 2. Analista SAP
- **Nível:** Pleno
- **Idiomas:** Inglês e Espanhol Fluentes
- **CV:** Experiência em SAP, SQL, Oracle, Excel
- **Resultado esperado:** Depende da compatibilidade com a vaga

## Troubleshooting

### Erro de Conexão
- Verifique se a API está rodando
- Confirme a URL base
- Verifique se não há firewall bloqueando

### Erro 500 - Modelo não carregado
- Verifique se o modelo está disponível em `models/`
- Confirme se todos os arquivos necessários estão presentes

### Erro 422 - Dados inválidos
- Verifique se os dados estão no formato correto
- Confirme se todos os campos obrigatórios estão preenchidos

## Logs e Monitoramento

A API registra todas as predições no MLflow para monitoramento:
- Acesse `/mlflow_info` para ver informações do experimento
- Use `/docs` para interface interativa de testes
- Logs detalhados são exibidos no console da API

## Deploy na Nuvem

Para testar na nuvem:
1. Faça deploy da API (Heroku, AWS, etc.)
2. Atualize a URL base nos scripts de teste
3. Execute os testes usando os scripts fornecidos
4. Monitore os logs para verificar o funcionamento

## Suporte

Se encontrar problemas:
1. Verifique os logs da API
2. Teste os endpoints básicos primeiro (`/health`, `/model_info`)
3. Use a documentação interativa em `/docs`
4. Verifique se todos os arquivos de modelo estão presentes
