#!/usr/bin/env python3
"""
Script para limpar espaços duplos deixados após remoção de emojis
"""

import os
import re
import glob

def cleanup_spaces(text):
    """Remove espaços duplos e limpa texto"""
    # Remover espaços duplos
    text = re.sub(r'  +', ' ', text)
    # Remover espaços no início de linhas
    text = re.sub(r'\n ', '\n', text)
    # Remover linhas vazias duplicadas
    text = re.sub(r'\n\n\n+', '\n\n', text)
    return text

def process_file(filepath):
    """Processa um arquivo limpando espaços"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Limpar espaços
        cleaned_content = cleanup_spaces(content)
        
        # Se houve mudança, salvar arquivo
        if cleaned_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Espaços limpos em: {filepath}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return False

def main():
    """Função principal"""
    print("Limpando espaços duplos dos arquivos...")
    
    # Extensões de arquivos para processar
    extensions = ['*.md', '*.py', '*.txt', '*.sh']
    
    files_changed = 0
    
    for extension in extensions:
        files = glob.glob(extension)
        for filepath in files:
            if process_file(filepath):
                files_changed += 1
    
    print(f"Arquivos modificados: {files_changed}")

if __name__ == "__main__":
    main()
