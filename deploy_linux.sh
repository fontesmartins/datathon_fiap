#!/bin/bash
# Script de Deploy para Linux
# Decision Recruitment AI

set -e  # Exit on any error

echo "🐧 DEPLOY DECISION RECRUITMENT AI - LINUX 🐧"
echo "=============================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar se Docker está instalado
check_docker() {
    log_info "Verificando se Docker está instalado..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker não está instalado!"
        echo "Instale o Docker:"
        echo "  Ubuntu/Debian: sudo apt-get update && sudo apt-get install docker.io"
        echo "  CentOS/RHEL: sudo yum install docker"
        echo "  Fedora: sudo dnf install docker"
        exit 1
    fi
    log_success "Docker está instalado"
}

# Verificar se Docker está rodando
check_docker_running() {
    log_info "Verificando se Docker está rodando..."
    if ! docker info &> /dev/null; then
        log_error "Docker não está rodando!"
        echo "Inicie o Docker:"
        echo "  sudo systemctl start docker"
        echo "  sudo systemctl enable docker"
        exit 1
    fi
    log_success "Docker está rodando"
}

# Verificar se modelo existe
check_model() {
    log_info "Verificando se o modelo está treinado..."
    if [ ! -f "models/xgboost_model.pkl" ]; then
        log_error "Modelo não encontrado!"
        echo "Execute primeiro: python main.py"
        exit 1
    fi
    log_success "Modelo encontrado"
}

# Parar container existente se estiver rodando
stop_existing_container() {
    log_info "Verificando containers existentes..."
    if docker ps -q -f name=decision-api | grep -q .; then
        log_warning "Container existente encontrado. Parando..."
        docker stop decision-api
        docker rm decision-api
        log_success "Container anterior removido"
    fi
}

# Construir imagem Docker
build_image() {
    log_info "Construindo imagem Docker..."
    docker build -t decision-recruitment-ai:latest .
    log_success "Imagem construída com sucesso"
}

# Executar container
run_container() {
    log_info "Iniciando container..."
    docker run -d \
        --name decision-api \
        -p 8000:8000 \
        --restart unless-stopped \
        decision-recruitment-ai:latest
    log_success "Container iniciado"
}

# Aguardar API estar pronta
wait_for_api() {
    log_info "Aguardando API estar pronta..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API está pronta!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    log_error "API não ficou pronta em 60 segundos"
    return 1
}

# Testar API
test_api() {
    log_info "Testando API..."
    
    # Teste de health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Health check OK"
    else
        log_error "Health check falhou"
        return 1
    fi
    
    # Teste de predição
    local test_response=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "candidate": {
                "nome": "Teste Linux",
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
        }' 2>/dev/null)
    
    if echo "$test_response" | grep -q "prediction"; then
        log_success "Teste de predição OK"
    else
        log_error "Teste de predição falhou"
        return 1
    fi
}

# Mostrar status final
show_status() {
    echo ""
    echo "=============================================="
    log_success "DEPLOY CONCLUÍDO COM SUCESSO!"
    echo "=============================================="
    echo ""
    echo "🌐 API disponível em: http://localhost:8000"
    echo "📚 Documentação: http://localhost:8000/docs"
    echo "🔍 Health check: http://localhost:8000/health"
    echo ""
    echo "📊 Comandos úteis:"
    echo "  docker ps                    # Ver containers rodando"
    echo "  docker logs decision-api     # Ver logs da API"
    echo "  docker stop decision-api     # Parar API"
    echo "  docker start decision-api    # Iniciar API"
    echo "  docker restart decision-api  # Reiniciar API"
    echo ""
    echo "🧪 Testar API:"
    echo "  curl http://localhost:8000/health"
    echo "  curl http://localhost:8000/model_info"
    echo ""
}

# Função principal
main() {
    echo "Iniciando deploy para Linux..."
    
    # Verificações
    check_docker
    check_docker_running
    check_model
    
    # Deploy
    stop_existing_container
    build_image
    run_container
    
    if wait_for_api; then
        if test_api; then
            show_status
        else
            log_error "Testes falharam"
            docker logs decision-api
            exit 1
        fi
    else
        log_error "API não ficou pronta"
        docker logs decision-api
        exit 1
    fi
}

# Executar se chamado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
