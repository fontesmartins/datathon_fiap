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
            self.jobs_data = json.load(f)
        print(f"✓ Vagas carregadas: {len(self.jobs_data)} registros")
        
        # Carregar prospecções
        with open(f'{self.data_path}/prospects.json', 'r', encoding='utf-8') as f:
            self.prospects_data = json.load(f)
        print(f"✓ Prospecções carregadas: {len(self.prospects_data)} registros")
        
        # Carregar candidatos (amostra para análise inicial)
        print("Carregando candidatos (arquivo grande)...")
        with open(f'{self.data_path}/applicants.json', 'r', encoding='utf-8') as f:
            self.applicants_data = json.load(f)
        print(f"✓ Candidatos carregados: {len(self.applicants_data)} registros")
        
    def analyze_data_structure(self):
        """Analisa a estrutura dos dados"""
        print("\n=== ANÁLISE DA ESTRUTURA DOS DADOS ===")
        
        # Análise das vagas
        print("\n1. ESTRUTURA DAS VAGAS:")
        sample_job = list(self.jobs_data.values())[0]
        print(f"   Campos principais: {list(sample_job.keys())}")
        
        # Análise das prospecções
        print("\n2. ESTRUTURA DAS PROSPECÇÕES:")
        sample_prospect = list(self.prospects_data.values())[0]
        print(f"   Campos principais: {list(sample_prospect.keys())}")
        if 'prospects' in sample_prospect:
            print(f"   Número de candidatos por vaga: {len(sample_prospect['prospects'])}")
        
        # Análise dos candidatos
        print("\n3. ESTRUTURA DOS CANDIDATOS:")
        sample_applicant = list(self.applicants_data.values())[0]
        print(f"   Campos principais: {list(sample_applicant.keys())}")
        
    def extract_features_from_jobs(self):
        """Extrai features das vagas"""
        print("\n=== EXTRAINDO FEATURES DAS VAGAS ===")
        
        jobs_features = []
        
        for job_id, job_data in self.jobs_data.items():
            features = {
                'job_id': job_id,
                'vaga_sap': job_data.get('informacoes_basicas', {}).get('vaga_sap', 'Não'),
                'cliente': job_data.get('informacoes_basicas', {}).get('cliente', ''),
                'nivel_profissional': job_data.get('perfil_vaga', {}).get('nivel profissional', ''),
                'nivel_academico': job_data.get('perfil_vaga', {}).get('nivel_academico', ''),
                'nivel_ingles': job_data.get('perfil_vaga', {}).get('nivel_ingles', ''),
                'nivel_espanhol': job_data.get('perfil_vaga', {}).get('nivel_espanhol', ''),
                'areas_atuacao': job_data.get('perfil_vaga', {}).get('areas_atuacao', ''),
                'cidade': job_data.get('perfil_vaga', {}).get('cidade', ''),
                'estado': job_data.get('perfil_vaga', {}).get('estado', ''),
                'titulo_vaga': job_data.get('informacoes_basicas', {}).get('titulo_vaga', ''),
                'tipo_contratacao': job_data.get('informacoes_basicas', {}).get('tipo_contratacao', ''),
                'data_requisicao': job_data.get('informacoes_basicas', {}).get('data_requicisao', ''),
                'principais_atividades': job_data.get('perfil_vaga', {}).get('principais_atividades', ''),
                'competencia_tecnicas': job_data.get('perfil_vaga', {}).get('competencia_tecnicas_e_comportamentais', '')
            }
            jobs_features.append(features)
        
        self.jobs_df = pd.DataFrame(jobs_features)
        print(f"✓ Features das vagas extraídas: {len(self.jobs_df)} registros")
        return self.jobs_df
    
    def extract_features_from_prospects(self):
        """Extrai features das prospecções"""
        print("\n=== EXTRAINDO FEATURES DAS PROSPECÇÕES ===")
        
        prospects_features = []
        
        for job_id, prospect_data in self.prospects_data.items():
            if 'prospects' in prospect_data:
                for prospect in prospect_data['prospects']:
                    features = {
                        'job_id': job_id,
                        'candidate_id': prospect.get('codigo', ''),
                        'candidate_name': prospect.get('nome', ''),
                        'situacao_candidato': prospect.get('situacao_candidado', ''),
                        'data_candidatura': prospect.get('data_candidatura', ''),
                        'ultima_atualizacao': prospect.get('ultima_atualizacao', ''),
                        'comentario': prospect.get('comentario', ''),
                        'recrutador': prospect.get('recrutador', '')
                    }
                    prospects_features.append(features)
        
        self.prospects_df = pd.DataFrame(prospects_features)
        print(f"✓ Features das prospecções extraídas: {len(self.prospects_df)} registros")
        return self.prospects_df
    
    def extract_features_from_applicants(self):
        """Extrai features dos candidatos"""
        print("\n=== EXTRAINDO FEATURES DOS CANDIDATOS ===")
        
        applicants_features = []
        
        for candidate_id, applicant_data in self.applicants_data.items():
            features = {
                'candidate_id': candidate_id,
                'nome': applicant_data.get('infos_basicas', {}).get('nome', ''),
                'email': applicant_data.get('infos_basicas', {}).get('email', ''),
                'telefone': applicant_data.get('infos_basicas', {}).get('telefone', ''),
                'data_criacao': applicant_data.get('infos_basicas', {}).get('data_criacao', ''),
                'data_atualizacao': applicant_data.get('infos_basicas', {}).get('data_atualizacao', ''),
                'inserido_por': applicant_data.get('infos_basicas', {}).get('inserido_por', ''),
                'objetivo_profissional': applicant_data.get('infos_basicas', {}).get('objetivo_profissional', ''),
                'local': applicant_data.get('infos_basicas', {}).get('local', ''),
                'data_nascimento': applicant_data.get('informacoes_pessoais', {}).get('data_nascimento', ''),
                'sexo': applicant_data.get('informacoes_pessoais', {}).get('sexo', ''),
                'estado_civil': applicant_data.get('informacoes_pessoais', {}).get('estado_civil', ''),
                'pcd': applicant_data.get('informacoes_pessoais', {}).get('pcd', ''),
                'titulo_profissional': applicant_data.get('informacoes_profissionais', {}).get('titulo_profissional', ''),
                'area_atuacao': applicant_data.get('informacoes_profissionais', {}).get('area_atuacao', ''),
                'conhecimentos_tecnicos': applicant_data.get('informacoes_profissionais', {}).get('conhecimentos_tecnicos', ''),
                'certificacoes': applicant_data.get('informacoes_profissionais', {}).get('certificacoes', ''),
                'remuneracao': applicant_data.get('informacoes_profissionais', {}).get('remuneracao', ''),
                'nivel_profissional': applicant_data.get('informacoes_profissionais', {}).get('nivel_profissional', ''),
                'nivel_academico': applicant_data.get('formacao_e_idiomas', {}).get('nivel_academico', ''),
                'nivel_ingles': applicant_data.get('formacao_e_idiomas', {}).get('nivel_ingles', ''),
                'nivel_espanhol': applicant_data.get('formacao_e_idiomas', {}).get('nivel_espanhol', ''),
                'cv_pt': applicant_data.get('cv_pt', ''),
                'cv_en': applicant_data.get('cv_en', '')
            }
            applicants_features.append(features)
        
        self.applicants_df = pd.DataFrame(applicants_features)
        print(f"✓ Features dos candidatos extraídas: {len(self.applicants_df)} registros")
        return self.applicants_df
    
    def create_unified_dataset(self):
        """Cria dataset unificado para análise preditiva"""
        print("\n=== CRIANDO DATASET UNIFICADO ===")
        
        # Merge das tabelas
        # 1. Prospects + Jobs
        merged = self.prospects_df.merge(
            self.jobs_df, 
            on='job_id', 
            how='left'
        )
        
        # 2. Adicionar dados dos candidatos
        merged = merged.merge(
            self.applicants_df, 
            on='candidate_id', 
            how='left'
        )
        
        self.unified_dataset = merged
        print(f"✓ Dataset unificado criado: {len(self.unified_dataset)} registros")
        
        return self.unified_dataset
    
    def analyze_target_variable(self):
        """Analisa a variável target (situação do candidato)"""
        print("\n=== ANÁLISE DA VARIÁVEL TARGET ===")
        
        if self.unified_dataset is not None:
            # Contar situações dos candidatos
            situacoes = self.unified_dataset['situacao_candidato'].value_counts()
            print("\nDistribuição das situações dos candidatos:")
            for situacao, count in situacoes.items():
                percentage = (count / len(self.unified_dataset)) * 100
                print(f"  {situacao}: {count} ({percentage:.1f}%)")
            
            # Identificar candidatos contratados
            contratados = self.unified_dataset[
                self.unified_dataset['situacao_candidato'].str.contains('Contratado', na=False)
            ]
            print(f"\n✓ Candidatos contratados: {len(contratados)}")
            
            return situacoes
    
    def feature_engineering(self):
        """Engenharia de features para o modelo preditivo"""
        print("\n=== ENGENHARIA DE FEATURES ===")
        
        df = self.unified_dataset.copy()
        
        # 1. Variável target binária
        df['contratado'] = df['situacao_candidato'].str.contains('Contratado', na=False).astype(int)
        
        # 2. Features de compatibilidade entre vaga e candidato
        df['nivel_profissional_match'] = (df['nivel_profissional_x'] == df['nivel_profissional_y']).astype(int)
        df['nivel_academico_match'] = (df['nivel_academico_x'] == df['nivel_academico_y']).astype(int)
        df['nivel_ingles_match'] = (df['nivel_ingles_x'] == df['nivel_ingles_y']).astype(int)
        df['nivel_espanhol_match'] = (df['nivel_espanhol_x'] == df['nivel_espanhol_y']).astype(int)
        
        # 3. Features de texto (comprimento do CV)
        df['cv_length'] = df['cv_pt'].str.len().fillna(0)
        df['has_cv_en'] = (df['cv_en'].str.len() > 0).astype(int)
        
        # 4. Features temporais
        df['data_candidatura'] = pd.to_datetime(df['data_candidatura'], errors='coerce')
        df['data_requisicao'] = pd.to_datetime(df['data_requisicao'], errors='coerce')
        df['dias_entre_requisicao_candidatura'] = (df['data_candidatura'] - df['data_requisicao']).dt.days
        
        # 5. Features categóricas
        df['is_sap_vaga'] = (df['vaga_sap'] == 'Sim').astype(int)
        df['is_pcd'] = (df['pcd'] == 'Sim').astype(int)
        
        # 6. Features de localização
        df['is_sp'] = df['estado'].str.contains('São Paulo', na=False).astype(int)
        
        # 7. Features de remuneração
        df['remuneracao_numeric'] = pd.to_numeric(df['remuneracao'], errors='coerce').fillna(0)
        
        self.features_df = df
        print(f"✓ Features criadas: {len(df.columns)} colunas")
        
        return df
    
    def get_model_features(self):
        """Retorna features selecionadas para o modelo"""
        feature_columns = [
            'is_sap_vaga',
            'nivel_profissional_match',
            'nivel_academico_match', 
            'nivel_ingles_match',
            'nivel_espanhol_match',
            'cv_length',
            'has_cv_en',
            'dias_entre_requisicao_candidatura',
            'is_pcd',
            'is_sp',
            'remuneracao_numeric'
        ]
        
        # Adicionar features categóricas codificadas
        categorical_features = [
            'cliente', 'nivel_profissional_x', 'nivel_academico_x',
            'nivel_ingles_x', 'nivel_espanhol_x', 'area_atuacao',
            'cidade', 'tipo_contratacao', 'titulo_profissional',
            'nivel_profissional_y', 'nivel_academico_y', 'nivel_ingles_y',
            'nivel_espanhol_y', 'recrutador'
        ]
        
        return feature_columns, categorical_features
    
    def generate_summary_report(self):
        """Gera relatório resumo da análise"""
        print("\n" + "="*60)
        print("RELATÓRIO RESUMO - ANÁLISE DE DADOS PARA MODELO PREDITIVO")
        print("="*60)
        
        print(f"\n📊 DADOS CARREGADOS:")
        print(f"   • Vagas: {len(self.jobs_data):,}")
        print(f"   • Prospecções: {len(self.prospects_data):,}")
        print(f"   • Candidatos: {len(self.applicants_data):,}")
        
        if hasattr(self, 'unified_dataset'):
            print(f"\n🔗 DATASET UNIFICADO:")
            print(f"   • Total de registros: {len(self.unified_dataset):,}")
            print(f"   • Colunas: {len(self.unified_dataset.columns)}")
            
            # Taxa de contratação
            if 'contratado' in self.unified_dataset.columns:
                taxa_contratacao = self.unified_dataset['contratado'].mean() * 100
                print(f"   • Taxa de contratação: {taxa_contratacao:.2f}%")
        
        print(f"\n🎯 VARIÁVEL TARGET:")
        if hasattr(self, 'unified_dataset'):
            situacoes = self.unified_dataset['situacao_candidato'].value_counts()
            for situacao, count in situacoes.head(5).items():
                print(f"   • {situacao}: {count:,}")
        
        print(f"\n🔧 FEATURES PARA MODELO:")
        feature_columns, categorical_features = self.get_model_features()
        print(f"   • Features numéricas: {len(feature_columns)}")
        print(f"   • Features categóricas: {len(categorical_features)}")
        
        print(f"\n✅ PRÓXIMOS PASSOS:")
        print(f"   1. Codificar variáveis categóricas")
        print(f"   2. Tratar valores faltantes")
        print(f"   3. Dividir em treino/teste")
        print(f"   4. Treinar modelo preditivo")
        print(f"   5. Avaliar performance")

def main():
    """Função principal"""
    data_path = "/Users/kielmartins/Desktop/code/dev/fiap-final"
    
    # Inicializar analisador
    analyzer = RecruitingDataAnalyzer(data_path)
    
    # Executar análise completa
    analyzer.load_data()
    analyzer.analyze_data_structure()
    analyzer.extract_features_from_jobs()
    analyzer.extract_features_from_prospects()
    analyzer.extract_features_from_applicants()
    analyzer.create_unified_dataset()
    analyzer.analyze_target_variable()
    analyzer.feature_engineering()
    analyzer.generate_summary_report()
    
    # Salvar dataset preparado
    if hasattr(analyzer, 'features_df'):
        output_file = f"{data_path}/dataset_preparado.csv"
        analyzer.features_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Dataset salvo em: {output_file}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
