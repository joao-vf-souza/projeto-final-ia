"""
app.py

Aplicação Streamlit para diagnóstico médico baseado em sintomas.
Utiliza modelo Random Forest treinado para prever diagnósticos e classificar nível de emergência.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from emergency_level import EmergencyLevel
import os
from PIL import Image
import joblib
import sys

# Importar a classe do modelo
sys.path.append(os.path.dirname(__file__))
from train_model_real import DiagnosticClassifierReal

# Configuração da página
st.set_page_config(
    page_title="Sistema de Diagnóstico Médico",
    page_icon="hospital",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .diagnosis-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .emergency-box-verde {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .emergency-box-amarelo {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .emergency-box-laranja {
        background-color: #ffe4cc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .emergency-box-vermelho {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Função para carregar o modelo treinado
@st.cache_resource
def load_model():
    """
    Carrega o modelo Random Forest treinado do disco.
    Utiliza cache do Streamlit para evitar recarregamento desnecessário.
    
    Retorna:
        DiagnosticClassifierReal: Modelo treinado
        
    Raises:
        Interrompe execução se modelo não for encontrado
    """
    model_path = 'data/model_real.pkl'
    
    # Verifica se o modelo existe
    if not os.path.exists(model_path):
        st.error("Modelo não encontrado!")
        st.info("""
        **Para usar este sistema:**
        
        1. Baixe o dataset do Kaggle: https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease
        2. Coloque o arquivo `Diseases_and_Symptoms_dataset.csv` na pasta `data/`
        3. Execute: `python train_model_real.py`
        4. Reinicie o Streamlit
        """)
        st.stop()
    
    # Carrega o modelo serializado
    try:
        classifier = joblib.load(model_path)
        return classifier
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        st.stop()

# Função para carregar descrições de doenças
@st.cache_data
def load_disease_descriptions():
    """
    Carrega descrições detalhadas das doenças do arquivo CSV.
    
    Retorna:
        dict: Dicionário {doença: descrição}
    """
    desc_path = 'data/description.csv'
    
    # Verifica se arquivo existe
    if not os.path.exists(desc_path):
        return {}
    
    try:
        df = pd.read_csv(desc_path)
        # Cria dicionário com primeira coluna como chave e segunda como valor
        descriptions = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return descriptions
    except Exception as e:
        return {}

# Função para renderizar box de emergência
def render_emergency_box(level_info):
    """
    Renderiza visualmente o nível de emergência do diagnóstico.
    
    Parâmetros:
        level_info (dict): Informações do nível de emergência
    """
    level = level_info['level']
    html_class = f"emergency-box-{level.lower()}"
    
    html_content = f"""
    <div class="{html_class}">
        <h3>{level_info['color']} {level_info['descricao']}</h3>
        <p><strong>Ação:</strong> {level_info['acao']}</p>
        <p><strong>Recomendação:</strong> {level_info['recomendacao']}</p>
        {'<p style="color: red;"><strong>' + level_info['aviso'] + '</strong></p>' if 'aviso' in level_info else ''}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

# Carregar modelo e descrições
classifier = load_model()
disease_descriptions = load_disease_descriptions()

# Layout da aplicação
st.title("Sistema de Diagnóstico Baseado em Sintomas")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Informações do Projeto")
    st.info(f"""
    **Objetivo:** Diagnóstico de doenças/condições baseado em sintomas
    
    **Técnica:** Classificação Multi-classe (Random Forest)
    
    **Dataset:** 96,088 amostras com 230 sintomas
    
    **Diagnósticos:** {len(classifier.diagnoses)} condições diferentes
    """)
    
    st.header("Aviso Importante")
    st.warning("""
    Este é um sistema educacional de **DEMONSTRAÇÃO**. 
    
    **NÃO substitui atendimento médico profissional!**
    
    Em caso de emergência, ligue **192** ou procure o pronto-socorro.
    """)

# Abas de navegação
tab1, tab2, tab3, tab4 = st.tabs(["Diagnóstico", "Métricas", "Informações", "Dados"])

# ========================= ABA 1: DIAGNÓSTICO =========================
with tab1:
    st.header("Selecione os Sintomas")
    
    # Dica para melhorar confiança
    st.info("💡 **Dica:** Quanto mais sintomas você selecionar, maior será a confiança do diagnóstico. Marque todos os sintomas que você está sentindo.")
    
    # Grid de sintomas
    col1, col2, col3 = st.columns(3)
    
    symptoms_selected = {}
    symptoms_list = classifier.symptoms_list
    
    for i, symptom in enumerate(symptoms_list):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        
        with col:
            symptoms_selected[symptom] = st.checkbox(symptom, key=f"symptom_{symptom}")
    
    st.markdown("---")
    
    # Botão de diagnóstico
    if st.button("🔍 Realizar Diagnóstico", key="diagnose_btn", use_container_width=True):
        
        # Validar se pelo menos um sintoma foi selecionado
        if not any(symptoms_selected.values()):
            st.error("❌ Selecione pelo menos um sintoma!")
        else:
            # Converter para formato do modelo
            symptoms_dict = {k: (1 if v else 0) for k, v in symptoms_selected.items()}
            
            # Realizar predição
            diagnosis, confidence, all_probabilities = classifier.predict(symptoms_dict)
            
            # Obter nível de emergência
            emergency_level = EmergencyLevel.get_level(diagnosis, confidence)
            
            # Armazenar no session state para exibição
            st.session_state.last_diagnosis = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probabilities': all_probabilities,
                'emergency_level': emergency_level,
                'symptoms': symptoms_selected
            }
    
    # Exibir resultados se existirem
    if 'last_diagnosis' in st.session_state:
        result = st.session_state.last_diagnosis
        
        st.markdown("---")
        st.header("Resultado do Diagnóstico")
        
        # Diagnóstico principal e confiança
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Diagnóstico Previsto")
            st.markdown(f"**{result['diagnosis']}**")
            
            # Exibir descrição da doença se disponível
            diagnosis_clean = result['diagnosis'].strip().lower()
            description_found = None
            
            # Tentar encontrar descrição (case-insensitive)
            for disease_name, desc in disease_descriptions.items():
                if disease_name.strip().lower() == diagnosis_clean:
                    description_found = desc
                    break
            
            if description_found:
                st.markdown("#### 📋 Sobre esta condição:")
                st.info(description_found)
        
        with col2:
            st.markdown(f"### Confiança")
            st.metric("Nível de Confiança", f"{result['confidence']:.0%}")
            
            # Mostrar top 3 diagnósticos alternativos
            st.markdown("#### Top 3 Diagnósticos:")
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for i, (disease, prob) in enumerate(sorted_probs[:3], 1):
                st.write(f"{i}. **{disease}**: {prob:.1%}")
        
        st.markdown("---")
        
        # Nível de emergência
        st.markdown("### Nível de Emergência")
        render_emergency_box(result['emergency_level'])
        
        st.markdown("---")
        
        # Gráfico de probabilidades
        st.markdown("### Probabilidades por Diagnóstico")
        
        probs_df = pd.DataFrame({
            'Diagnóstico': list(result['probabilities'].keys()),
            'Probabilidade': list(result['probabilities'].values())
        }).sort_values('Probabilidade', ascending=False)
        
        fig = px.bar(
            probs_df,
            x='Diagnóstico',
            y='Probabilidade',
            color='Probabilidade',
            color_continuous_scale='RdYlGn',
            labels={'Probabilidade': 'Probabilidade'},
            title='Probabilidade de cada diagnóstico'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Resumo de sintomas selecionados
        st.markdown("### Sintomas Informados")
        sintomas_sim = [s for s, v in result['symptoms'].items() if v]
        if sintomas_sim:
            cols = st.columns(3)
            for i, symptom in enumerate(sintomas_sim):
                cols[i % 3].success(f"✓ {symptom}")

# ========================= ABA 2: MÉTRICAS =========================
with tab2:
    st.header("Métricas do Modelo")
    
    # Exibir informações básicas do modelo
    st.markdown("### Informações do Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sintomas", len(classifier.symptoms_list))
    col2.metric("Doenças", len(classifier.diagnoses))
    col3.metric("Tipo", "Random Forest")
    
    # Usar métricas salvas no modelo (do treinamento original)
    if hasattr(classifier, 'metrics') and classifier.metrics:
        st.markdown("### Desempenho do Modelo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Acurácia (Treino)", f"{classifier.metrics['accuracy_train']:.1%}")
        col2.metric("Acurácia (Teste)", f"{classifier.metrics['accuracy_test']:.1%}")
        col3.metric("Precisão", f"{classifier.metrics['precision']:.1%}")
        col4.metric("Recall", f"{classifier.metrics['recall']:.1%}")
    else:
        st.info("⚠️ Métricas não disponíveis. Retreine o modelo com `python train_model_real.py`")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### Importância das Features (Sintomas)")
    
    feature_importance = classifier.get_feature_importance()
    feature_df = pd.DataFrame({
        'Sintoma': list(feature_importance.keys()),
        'Importância': list(feature_importance.values())
    }).sort_values('Importância', ascending=False).head(20)
    
    fig = px.bar(
        feature_df,
        x='Importância',
        y='Sintoma',
        orientation='h',
        color='Importância',
        color_continuous_scale='Viridis',
        title='Top 20 Sintomas Mais Importantes'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Distribuição de diagnósticos
    dataset_path = 'data/Diseases_and_Symptoms_dataset.csv'
    if os.path.exists(dataset_path):
        st.markdown("### Distribuição de Diagnósticos no Dataset")
        df = pd.read_csv(dataset_path)
        diag_counts = df.iloc[:, 0].value_counts().head(20)
        fig = px.bar(
            x=diag_counts.values,
            y=diag_counts.index,
            orientation='h',
            title='Top 20 Diagnósticos Mais Frequentes',
            labels={'x': 'Número de Amostras', 'y': 'Diagnóstico'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ========================= ABA 3: INFORMAÇÕES =========================
with tab3:
    st.header("Informações sobre o Projeto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Objetivos")
        st.markdown("""
        - Diagnosticar condições médicas baseado em sintomas
        - Classificar nível de emergência
        - Fornecer recomendações apropriadas
        - Demonstrar aplicação de Machine Learning
        """)
        
        st.markdown("### Técnicas Utilizadas")
        st.markdown("""
        - **Algoritmo:** Random Forest Classifier
        - **Tipo:** Classificação Multi-classe
        - **Framework:** Scikit-learn
        - **Interface:** Streamlit
        """)
    
    with col2:
        st.markdown("### Dataset")
        st.markdown(f"""
        - **Tamanho:** {len(classifier.diagnoses) * 900:,} amostras
        - **Features:** {len(classifier.symptoms_list)} sintomas (binários)
        - **Classes:** {len(classifier.diagnoses)} diagnósticos
        - **Fonte:** SymScan - Kaggle Dataset
        """)
        
        st.markdown("### Níveis de Emergência")
        st.markdown("""
        - **🟢 Verde:** Emergência baixa (consulta em dias)
        - **🟡 Amarelo:** Urgência (consulta em poucas horas)
        - **🟠 Laranja:** Emergência (procurar ER hoje)
        - **🔴 Vermelho:** Crítica (ambulância imediato)
        """)
    
    st.markdown("---")
    
    st.markdown("### Aviso de Saúde")
    st.warning("""
    **ESTE SISTEMA É APENAS PARA FINS EDUCACIONAIS**
    
    - Não substitui diagnóstico médico profissional
    - Não deve ser usado como base para decisões de tratamento
    - Em caso de emergência, ligue **192** ou procure o pronto-socorro
    - Consulte sempre um médico qualificado
    """)
    
    st.markdown("### Sobre a Implementação")
    st.markdown("""
    **Stack Tecnológico:**
    - Python 3.8+
    - Scikit-learn (ML)
    - Streamlit (Interface)
    - Plotly (Visualizações)
    - Pandas/NumPy (Dados)
    
    **Arquitetura:**
    - `classifier.py`: Modelo de classificação
    - `emergency_level.py`: Sistema de nível de emergência
    - `app.py`: Interface Streamlit
    """)

# ========================= ABA 4: DADOS =========================
with tab4:
    st.header("Dados do Modelo")
    
    dataset_path = 'data/Diseases_and_Symptoms_dataset.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        st.markdown("### Dataset Completo")
        
        # Nome da primeira coluna (diagnóstico)
        diagnosis_col = df.columns[0]
        
        # Filtro por diagnóstico
        diagnosis_filter = st.multiselect(
            "Filtrar por diagnóstico:",
            options=df[diagnosis_col].unique(),
            default=df[diagnosis_col].unique()[:2]
        )
        
        df_filtered = df[df[diagnosis_col].isin(diagnosis_filter)]
        
        st.dataframe(df_filtered, use_container_width=True, height=400)
        
        st.markdown(f"**Total de linhas:** {len(df_filtered)} / {len(df)}")
        
        # Estatísticas
        st.markdown("---")
        st.markdown("### Estatísticas")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Amostras", len(df))
        col2.metric("Número de Sintomas", len(classifier.symptoms_list))
        col3.metric("Número de Diagnósticos", df[diagnosis_col].nunique())
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv,
            file_name="symptoms_dataset.csv",
            mime="text/csv"
        )
    else:
        st.info("Dataset ainda não foi criado.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>🏥 Sistema de Diagnóstico com Nível de Emergência | Trabalho Final de Inteligência Artificial</p>
    <p>Desenvolvido com Streamlit, Scikit-learn e Python</p>
</div>
""", unsafe_allow_html=True)