#!/usr/bin/env python3
"""
Script para remover emojis de todos os arquivos do projeto
"""

import os
import re
import glob

def remove_emojis_from_text(text):
    """Remove emojis de um texto"""
    # Padrão para detectar emojis (Unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def process_file(filepath):
    """Processa um arquivo removendo emojis"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remover emojis
        cleaned_content = remove_emojis_from_text(content)
        
        # Se houve mudança, salvar arquivo
        if cleaned_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Emojis removidos de: {filepath}")
            return True
        else:
            print(f"Nenhum emoji encontrado em: {filepath}")
            return False
            
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return False

def main():
    """Função principal"""
    print("Removendo emojis de todos os arquivos do projeto...")
    
    # Extensões de arquivos para processar
    extensions = ['*.md', '*.py', '*.txt', '*.sh', '*.json']
    
    files_processed = 0
    files_changed = 0
    
    for extension in extensions:
        files = glob.glob(extension)
        for filepath in files:
            files_processed += 1
            if process_file(filepath):
                files_changed += 1
    
    print(f"\nProcessamento concluído:")
    print(f"Arquivos processados: {files_processed}")
    print(f"Arquivos modificados: {files_changed}")

if __name__ == "__main__":
    main()
