# 🏥 Sistema de Diagnóstico Médico com Inteligência Artificial

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://joao-vf-souza-projeto-final-ia-app-6ysln1.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)

> Sistema inteligente de diagnóstico médico preliminar baseado em Machine Learning, desenvolvido como projeto final do curso de Inteligência Artificial - Bacharelado em Sistemas de Informação.

## 🚀 Demonstração

**Acesse a aplicação online:** [https://joao-vf-souza-projeto-final-ia-app-6ysln1.streamlit.app/](https://joao-vf-souza-projeto-final-ia-app-6ysln1.streamlit.app/)

![Sistema em Ação](https://img.shields.io/badge/Status-Online-success)

## 📋 Sobre o Projeto

Este projeto implementa um sistema automatizado de diagnóstico médico que utiliza algoritmos de Machine Learning para prever diagnósticos a partir de sintomas reportados pelo usuário. O sistema foi desenvolvido como trabalho final da disciplina de Inteligência Artificial e demonstra a aplicação prática de técnicas de classificação multi-classe em um problema real do domínio da saúde.

### 🎯 Principais Funcionalidades

- **Diagnóstico Inteligente**: Classifica 100 diferentes condições médicas com base em 230 sintomas
- **Sistema de Emergência**: Classifica automaticamente o nível de urgência (Verde, Amarelo, Laranja, Vermelho)
- **Interface Interativa**: Aplicação web responsiva desenvolvida com Streamlit
- **Análise de Confiança**: Exibe probabilidades e alternativas diagnósticas (confiança típica: 20-40%)
- **Visualizações**: Gráficos interativos de importância de features e distribuição de dados
- **Métricas Transparentes**: Acurácia de 89.22%, Precisão de 91.30%, Recall de 89.22%
- **Descrições de Doenças**: Informações detalhadas sobre cada condição diagnosticada

## 🧠 Tecnologia e Metodologia

### Algoritmo Utilizado

**Random Forest Classifier** com hiperparâmetros otimizados:
- 300 árvores de decisão
- Profundidade máxima de 40
- Técnicas de regularização (pruning, bagging)
- Balanceamento automático de classes

### Dataset

- **Nome**: SymScan - Symptoms to Disease Dataset
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)
- **Amostras**: 96.088 registros
- **Features**: 230 sintomas binários
- **Classes**: 100 diagnósticos diferentes
- **Distribuição**: Balanceada (~960 amostras/classe)

### Métricas de Desempenho

| Métrica | Treino | Teste |
|---------|--------|-------|
| Acurácia | 88.90% | **89.22%** |
| Precisão | - | **91.30%** |
| Recall | - | **89.22%** |

> ✅ **Destaque**: Acurácia de teste superior à de treino, indicando excelente capacidade de generalização sem overfitting.

## 🛠️ Stack Tecnológica

- **Linguagem**: Python 3.11+
- **Machine Learning**: scikit-learn 1.7.2
- **Interface Web**: Streamlit 1.51.0
- **Manipulação de Dados**: Pandas 2.2.3, NumPy 1.26.4
- **Visualizações**: Plotly 5.17.0, Matplotlib 3.8.1
- **Serialização**: Joblib 1.3.2

## 📦 Instalação e Uso

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)

### Instalação Local

```bash
# Clone o repositório
git clone https://github.com/joao-vf-souza/projeto-final-ia.git
cd projeto-final-ia

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

### Treinamento do Modelo (Opcional)

Se você quiser re-treinar o modelo do zero:

```bash
# 1. Baixe o dataset do Kaggle
# https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease

# 2. Coloque o arquivo Diseases_and_Symptoms_dataset.csv na pasta data/

# 3. Execute o treinamento
python train_model_real.py
```

> **Nota**: O modelo treinado já está incluído no repositório (`data/model_real.pkl`), então este passo é opcional.

## 📁 Estrutura do Projeto

```
projeto-final-ia/
├── app.py                                  # Interface Streamlit
├── train_model_real.py                     # Script de treinamento do modelo
├── emergency_level.py                      # Sistema de níveis de emergência (4 níveis)
├── requirements.txt                        # Dependências do projeto
├── README.md                               # Este arquivo
├── .gitignore                              # Arquivos ignorados pelo Git
└── data/
    ├── Diseases_and_Symptoms_dataset.csv  # Dataset principal (96.088 amostras)
    ├── description.csv                     # Descrições detalhadas das 100 doenças
    └── model_real.pkl                      # Modelo treinado (~50MB)
```

## 🎓 Contexto Acadêmico

**Instituição**: UNESP - Universidade Estadual Paulista  
**Campus**: Bauru  
**Curso**: Bacharelado em Sistemas de Informação  
**Disciplina**: Inteligência Artificial  
**Professor**: Clayton Pereira  
**Data de Entrega**: 01/12/2025  

## 📊 Funcionalidades da Interface

### 1️⃣ Aba Diagnóstico
- Seleção de sintomas via checkboxes
- Diagnóstico em tempo real
- Nível de confiança da predição
- Top 3 diagnósticos alternativos
- Classificação de emergência com recomendações
- Gráfico de probabilidades

### 2️⃣ Aba Métricas
- Informações do modelo treinado
- Métricas de desempenho detalhadas
- Top 20 sintomas mais importantes
- Distribuição de diagnósticos no dataset

### 3️⃣ Aba Informações
- Documentação do modelo e metodologia
- Detalhes do dataset utilizado
- Avisos de uso educacional
- Stack tecnológica completa

### 4️⃣ Aba Dados
- Visualização do dataset completo
- Filtros por diagnóstico
- Estatísticas descritivas
- Download em formato CSV

## 📊 Sobre a Confiança das Predições

Devido ao grande número de classes (100 doenças), é **normal e esperado** que o modelo apresente níveis de confiança relativamente baixos:

- **Confiança Típica**: 20-40%
- **Confiança Máxima Observada**: ~75%
- **Confiança Média**: ~28%

Esta característica é inerente a problemas de classificação multi-classe com muitas opções. O modelo ainda é altamente preciso (91.30% de precisão), mas distribui probabilidades entre múltiplas doenças similares.

### Sistema de Avisos de Confiança

- **Confiança ≥ 40%**: Sem aviso adicional
- **Confiança 25-40%**: "Confiança moderada - Recomenda-se consulta médica"
- **Confiança < 25%**: "Confiança muito baixa - CONSULTE UM MÉDICO"

## 🚨 Sistema de Níveis de Emergência

O sistema classifica automaticamente a urgência do diagnóstico em **4 níveis** baseado em mapeamento de 100 doenças:

| Nível | Cor | Descrição | Recomendação |
|-------|-----|-----------|--------------|
| 🟢 **Verde** | Baixo | Emergência Baixa | Agendar consulta em dias |
| 🟡 **Amarelo** | Moderado | Urgência | Procurar UPA em horas |
| 🟠 **Laranja** | Alto | Emergência | Procurar pronto-socorro hoje |
| 🔴 **Vermelho** | Crítico | Risco de Vida | Ligar 192 (SAMU) imediatamente |

## ⚠️ Avisos Importantes

> **⚠️ ATENÇÃO**: Este sistema foi desenvolvido exclusivamente para fins educacionais e demonstração acadêmica.

- ❌ **NÃO** substitui consulta médica profissional
- ❌ **NÃO** deve ser usado para decisões médicas reais
- ✅ Ferramenta de aprendizado sobre Machine Learning aplicado à saúde
- ✅ Em caso de emergência real, procure atendimento médico qualificado

## 📚 Documentação Técnica

Documentação técnica detalhada disponível no repositório, incluindo:

- **Fundamentação teórica**: Random Forest e classificação multi-classe
- **Hiperparâmetros otimizados**: 300 árvores, profundidade 40, regularização
- **Pipeline completo**: Pré-processamento, treinamento, avaliação
- **Sistema de emergência**: Mapeamento de 100 doenças em 4 níveis
- **Análise de confiança**: Entendendo probabilidades em classificação multi-classe
- **Métricas de desempenho**: Acurácia, Precisão, Recall detalhados

## 🔮 Trabalhos Futuros

- Implementação multilíngue (português)
- Exploração de Deep Learning
- Incorporação de features adicionais (idade, sexo, histórico)
- API REST para integração com outros sistemas
- Aplicativo mobile
- Explicabilidade com SHAP/LIME
- Histórico de consultas por usuário

## 👤 Autores

**João Victor Fernandes Souza** e
**Vinicius Henrique de Oliveira Franzote**

## 🤝 Contribuições

Contribuições, issues e feature requests são bem-vindos! Sinta-se livre para abrir uma issue ou pull request.

## 🙏 Agradecimentos

- Prof. Clayton Pereira pela orientação na disciplina
- Comunidade Kaggle pelo dataset de qualidade
- Comunidade open-source pelas bibliotecas utilizadas

---

<div align="center">

**Desenvolvido com ❤️ para aprendizado e demonstração acadêmica**

[![UNESP](https://img.shields.io/badge/UNESP-Bauru-blue)](https://www.fc.unesp.br/)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)

**Dezembro/2025**

</div>
