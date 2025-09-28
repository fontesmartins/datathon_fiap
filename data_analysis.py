#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de Dados para Modelo Preditivo de Recrutamento
Sistema ATS - Análise de candidatos, vagas e prospecções
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class RecruitingDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.jobs_data = None
        self.prospects_data = None
        self.applicants_data = None
        self.unified_dataset = None

    def load_data(self):
        """Carrega todos os arquivos JSON"""
        print("Carregando dados...")
        
        # Carregar vagas
        with open(f'{self.data_path}/vagas.json', 'r', encoding='utf-8') as f:
            self.jobs_data = pd.DataFrame(json.load(f))
        
        # Carregar prospecções
        with open(f'{self.data_path}/prospects.json', 'r', encoding='utf-8') as f:
            self.prospects_data = pd.DataFrame(json.load(f))
        
        # Carregar candidatos
        with open(f'{self.data_path}/applicants.json', 'r', encoding='utf-8') as f:
            self.applicants_data = pd.DataFrame(json.load(f))
        
        print(f"✅ Vagas carregadas: {len(self.jobs_data)} registros")
        print(f"✅ Prospecções carregadas: {len(self.prospects_data)} registros")
        print(f"✅ Candidatos carregados: {len(self.applicants_data)} registros")

    def analyze_jobs_data(self):
        """Análise dos dados de vagas"""
        print("\n" + "="*50)
        print("ANÁLISE DOS DADOS DE VAGAS")
        print("="*50)
        
        print(f"\n📊 Resumo dos dados:")
        print(f"Total de vagas: {len(self.jobs_data)}")
        print(f"Colunas disponíveis: {list(self.jobs_data.columns)}")
        
        # Análise de tipos de contratação
        if 'tipo_contratacao' in self.jobs_data.columns:
            print(f"\n💼 Tipos de contratação:")
            contract_types = self.jobs_data['tipo_contratacao'].value_counts()
            for contract_type, count in contract_types.head(10).items():
                percentage = (count / len(self.jobs_data)) * 100
                print(f"  {contract_type}: {count} ({percentage:.1f}%)")
        
        # Análise de níveis profissionais
        if 'nivel_profissional' in self.jobs_data.columns:
            print(f"\n👔 Níveis profissionais:")
            levels = self.jobs_data['nivel_profissional'].value_counts()
            for level, count in levels.items():
                percentage = (count / len(self.jobs_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # Análise de idiomas
        if 'nivel_ingles' in self.jobs_data.columns:
            print(f"\n🇺🇸 Níveis de inglês:")
            english_levels = self.jobs_data['nivel_ingles'].value_counts()
            for level, count in english_levels.items():
                percentage = (count / len(self.jobs_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")
        
        if 'nivel_espanhol' in self.jobs_data.columns:
            print(f"\n🇪🇸 Níveis de espanhol:")
            spanish_levels = self.jobs_data['nivel_espanhol'].value_counts()
            for level, count in spanish_levels.items():
                percentage = (count / len(self.jobs_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")

    def analyze_prospects_data(self):
        """Análise dos dados de prospecções"""
        print("\n" + "="*50)
        print("ANÁLISE DOS DADOS DE PROSPECÇÕES")
        print("="*50)
        
        print(f"\n📊 Resumo dos dados:")
        print(f"Total de prospecções: {len(self.prospects_data)}")
        print(f"Colunas disponíveis: {list(self.prospects_data.columns)}")
        
        # Análise de status
        if 'status' in self.prospects_data.columns:
            print(f"\n📈 Status das prospecções:")
            status_counts = self.prospects_data['status'].value_counts()
            for status, count in status_counts.items():
                percentage = (count / len(self.prospects_data)) * 100
                print(f"  {status}: {count} ({percentage:.1f}%)")
        
        # Análise de fontes
        if 'fonte' in self.prospects_data.columns:
            print(f"\n🔍 Fontes das prospecções:")
            sources = self.prospects_data['fonte'].value_counts()
            for source, count in sources.head(10).items():
                percentage = (count / len(self.prospects_data)) * 100
                print(f"  {source}: {count} ({percentage:.1f}%)")

    def analyze_applicants_data(self):
        """Análise dos dados de candidatos"""
        print("\n" + "="*50)
        print("ANÁLISE DOS DADOS DE CANDIDATOS")
        print("="*50)
        
        print(f"\n📊 Resumo dos dados:")
        print(f"Total de candidatos: {len(self.applicants_data)}")
        print(f"Colunas disponíveis: {list(self.applicants_data.columns)}")
        
        # Análise de localização
        if 'cidade' in self.applicants_data.columns:
            print(f"\n🌍 Top 10 cidades dos candidatos:")
            cities = self.applicants_data['cidade'].value_counts()
            for city, count in cities.head(10).items():
                percentage = (count / len(self.applicants_data)) * 100
                print(f"  {city}: {count} ({percentage:.1f}%)")
        
        # Análise de níveis profissionais
        if 'nivel_profissional' in self.applicants_data.columns:
            print(f"\n👔 Níveis profissionais dos candidatos:")
            levels = self.applicants_data['nivel_profissional'].value_counts()
            for level, count in levels.items():
                percentage = (count / len(self.applicants_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # Análise de idiomas
        if 'nivel_ingles' in self.applicants_data.columns:
            print(f"\n🇺🇸 Níveis de inglês dos candidatos:")
            english_levels = self.applicants_data['nivel_ingles'].value_counts()
            for level, count in english_levels.items():
                percentage = (count / len(self.applicants_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")
        
        if 'nivel_espanhol' in self.applicants_data.columns:
            print(f"\n🇪🇸 Níveis de espanhol dos candidatos:")
            spanish_levels = self.applicants_data['nivel_espanhol'].value_counts()
            for level, count in spanish_levels.items():
                percentage = (count / len(self.applicants_data)) * 100
                print(f"  {level}: {count} ({percentage:.1f}%)")

    def create_unified_dataset(self):
        """Cria dataset unificado para análise"""
        print("\n" + "="*50)
        print("CRIAÇÃO DO DATASET UNIFICADO")
        print("="*50)
        
        # Mesclar dados de vagas e prospecções
        print("Mesclando dados de vagas e prospecções...")
        merged_data = pd.merge(
            self.prospects_data,
            self.jobs_data,
            left_on='vaga_id',
            right_on='id',
            how='inner'
        )
        print(f"✅ Dados mesclados: {len(merged_data)} registros")
        
        # Mesclar com dados de candidatos
        print("Mesclando com dados de candidatos...")
        self.unified_dataset = pd.merge(
            merged_data,
            self.applicants_data,
            left_on='candidato_id',
            right_on='id',
            how='inner'
        )
        print(f"✅ Dataset unificado criado: {len(self.unified_dataset)} registros")
        
        # Adicionar coluna de target (contratado)
        if 'status' in self.unified_dataset.columns:
            self.unified_dataset['contratado'] = (
                self.unified_dataset['status'] == 'contratado'
            ).astype(int)
            
            contratados = self.unified_dataset['contratado'].sum()
            total = len(self.unified_dataset)
            taxa_contratacao = (contratados / total) * 100
            
            print(f"\n🎯 Taxa de contratação: {contratados}/{total} ({taxa_contratacao:.2f}%)")

    def analyze_unified_dataset(self):
        """Análise do dataset unificado"""
        print("\n" + "="*50)
        print("ANÁLISE DO DATASET UNIFICADO")
        print("="*50)
        
        if self.unified_dataset is None:
            print("❌ Dataset unificado não foi criado ainda!")
            return
        
        print(f"\n📊 Resumo geral:")
        print(f"Total de registros: {len(self.unified_dataset)}")
        print(f"Total de colunas: {len(self.unified_dataset.columns)}")
        
        # Análise do target
        if 'contratado' in self.unified_dataset.columns:
            contratados = self.unified_dataset['contratado'].sum()
            nao_contratados = len(self.unified_dataset) - contratados
            
            print(f"\n🎯 Distribuição do target:")
            print(f"  Contratados: {contratados} ({(contratados/len(self.unified_dataset)*100):.2f}%)")
            print(f"  Não contratados: {nao_contratados} ({(nao_contratados/len(self.unified_dataset)*100):.2f}%)")
        
        # Análise de compatibilidade
        self.analyze_compatibility()
        
        # Análise de padrões
        self.analyze_patterns()

    def analyze_compatibility(self):
        """Análise de compatibilidade candidato-vaga"""
        print(f"\n🔍 Análise de compatibilidade:")
        
        # Compatibilidade de nível profissional
        if all(col in self.unified_dataset.columns for col in ['nivel_profissional_x', 'nivel_profissional_y']):
            nivel_match = (
                self.unified_dataset['nivel_profissional_x'] == 
                self.unified_dataset['nivel_profissional_y']
            )
            match_rate = nivel_match.mean() * 100
            print(f"  Compatibilidade nível profissional: {match_rate:.1f}%")
        
        # Compatibilidade de inglês
        if all(col in self.unified_dataset.columns for col in ['nivel_ingles_x', 'nivel_ingles_y']):
            ingles_match = (
                self.unified_dataset['nivel_ingles_x'] == 
                self.unified_dataset['nivel_ingles_y']
            )
            match_rate = ingles_match.mean() * 100
            print(f"  Compatibilidade inglês: {match_rate:.1f}%")
        
        # Compatibilidade de espanhol
        if all(col in self.unified_dataset.columns for col in ['nivel_espanhol_x', 'nivel_espanhol_y']):
            espanhol_match = (
                self.unified_dataset['nivel_espanhol_x'] == 
                self.unified_dataset['nivel_espanhol_y']
            )
            match_rate = espanhol_match.mean() * 100
            print(f"  Compatibilidade espanhol: {match_rate:.1f}%")

    def analyze_patterns(self):
        """Análise de padrões nos dados"""
        print(f"\n📈 Análise de padrões:")
        
        if 'contratado' not in self.unified_dataset.columns:
            return
        
        # Análise por tipo de contratação
        if 'tipo_contratacao' in self.unified_dataset.columns:
            print(f"\n💼 Taxa de contratação por tipo:")
            for contract_type in self.unified_dataset['tipo_contratacao'].unique():
                subset = self.unified_dataset[
                    self.unified_dataset['tipo_contratacao'] == contract_type
                ]
                if len(subset) > 10:  # Apenas tipos com mais de 10 registros
                    rate = subset['contratado'].mean() * 100
                    print(f"  {contract_type}: {rate:.1f}% ({len(subset)} registros)")
        
        # Análise por nível profissional
        if 'nivel_profissional_x' in self.unified_dataset.columns:
            print(f"\n👔 Taxa de contratação por nível:")
            for level in self.unified_dataset['nivel_profissional_x'].unique():
                subset = self.unified_dataset[
                    self.unified_dataset['nivel_profissional_x'] == level
                ]
                if len(subset) > 10:
                    rate = subset['contratado'].mean() * 100
                    print(f"  {level}: {rate:.1f}% ({len(subset)} registros)")

    def save_analysis_report(self, output_path="analysis_report.txt"):
        """Salva relatório de análise"""
        print(f"\n💾 Salvando relatório de análise em {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE DADOS - DECISION RECRUITMENT AI\n")
            f.write("="*60 + "\n\n")
            f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumo dos dados
            f.write("RESUMO DOS DADOS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total de vagas: {len(self.jobs_data) if self.jobs_data is not None else 0}\n")
            f.write(f"Total de prospecções: {len(self.prospects_data) if self.prospects_data is not None else 0}\n")
            f.write(f"Total de candidatos: {len(self.applicants_data) if self.applicants_data is not None else 0}\n")
            
            if self.unified_dataset is not None:
                f.write(f"Total de registros unificados: {len(self.unified_dataset)}\n")
                if 'contratado' in self.unified_dataset.columns:
                    contratados = self.unified_dataset['contratado'].sum()
                    taxa = (contratados / len(self.unified_dataset)) * 100
                    f.write(f"Taxa de contratação: {taxa:.2f}%\n")
        
        print(f"✅ Relatório salvo em {output_path}")

    def run_full_analysis(self):
        """Executa análise completa"""
        print("🚀 INICIANDO ANÁLISE COMPLETA DOS DADOS")
        print("="*60)
        
        try:
            # Carregar dados
            self.load_data()
            
            # Análises individuais
            self.analyze_jobs_data()
            self.analyze_prospects_data()
            self.analyze_applicants_data()
            
            # Dataset unificado
            self.create_unified_dataset()
            self.analyze_unified_dataset()
            
            # Salvar relatório
            self.save_analysis_report()
            
            print("\n" + "="*60)
            print("✅ ANÁLISE COMPLETA FINALIZADA!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Erro durante a análise: {e}")
            raise

def main():
    """Função principal"""
    print("Decision Recruitment AI - Análise de Dados")
    print("="*50)
    
    # Caminho dos dados (assumindo que estão na raiz do projeto)
    data_path = "."
    
    # Criar analisador e executar análise
    analyzer = RecruitingDataAnalyzer(data_path)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
