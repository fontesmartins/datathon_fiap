# Deploy no Railway.app

## Passos para Deploy

### 1. Preparação
```bash
# Garantir que o modelo está treinado
python main.py

# Verificar se todos os arquivos estão presentes
python check_models.py

# Testar localmente
python start.py
```

### 2. Deploy no Railway

1. **Acesse [Railway.app](https://railway.app)**
2. **Login com GitHub**
3. **New Project → Deploy from GitHub repo**
4. **Selecione seu repositório**
5. **Railway detectará automaticamente:**
   - `Dockerfile` (usará Docker)
   - `requirements.txt` (dependências Python)
   - `railway.json` (configurações)
   - `Procfile` (comando de start)

### 3. Configurações Railway

- **Port**: Railway define automaticamente via `$PORT`
- **Health Check**: `/health` endpoint
- **Auto Deploy**: Sim (a cada push no GitHub)

### 4. Variáveis de Ambiente (se necessário)

```bash
# No Railway Dashboard → Variables
PYTHONPATH=/app
MLFLOW_TRACKING_URI=file:./mlruns
```

### 5. Verificação

```bash
# Após deploy, teste:
curl https://seu-projeto.railway.app/health
curl https://seu-projeto.railway.app/model_info
```

## Arquivos Necessários

✅ `Dockerfile` - Configuração container
✅ `requirements.txt` - Dependências Python  
✅ `railway.json` - Configurações Railway
✅ `Procfile` - Comando de start
✅ `start.py` - Script de inicialização robusto
✅ `fastapi_app.py` - API principal
✅ `models/` - Modelos treinados
✅ `dataset_preparado.csv` - Dataset
✅ `check_models.py` - Verificador de modelos

## Limitações Railway

- **Storage**: Volátil (dados se perdem no redeploy)
- **MLflow**: Logs locais (não persistentes)
- **Modelos**: Devem estar no repo ou rebuild

## Solução para Persistência

Para manter modelos e logs:

1. **Commit modelos treinados**:
```bash
git add models/
git commit -m "Add trained models"
git push
```

2. **Ou usar Railway Volume** (pago):
```json
{
  "volumes": [
    {
      "mountPath": "/app/mlruns",
      "size": "1GB"
    }
  ]
}
```

## Monitoramento

- **Logs**: Railway Dashboard → Logs
- **Métricas**: Railway Dashboard → Metrics
- **Health**: Endpoint `/health`

## Custos

- **Free Tier**: $5 créditos/mês
- **Pro**: $20/mês (volumes persistentes)
- **Enterprise**: Customizado

---

**Railway é ideal para este projeto! 🚂**
