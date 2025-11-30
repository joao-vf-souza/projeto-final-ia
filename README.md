# Sistema de Diagnóstico Médico com IA

Sistema inteligente de diagnóstico baseado em sintomas usando Machine Learning. Desenvolvido como trabalho final do curso de Inteligência Artificial.

## Sobre o Projeto

Este sistema utiliza **Random Forest Classifier** para prever possíveis diagnósticos médicos baseado em sintomas informados pelo usuário. O modelo foi treinado com o dataset **SymScan** do Kaggle, contendo 96.088 amostras com 230 sintomas diferentes e 100 doenças.

### Funcionalidades

- ✅ Diagnóstico baseado em sintomas selecionados
- ✅ Classificação de nível de emergência (Verde, Amarelo, Laranja, Vermelho)
- ✅ Visualização de confiança e probabilidades
- ✅ Interface web interativa com Streamlit
- ✅ Análise de importância de sintomas
- ✅ Métricas e gráficos de desempenho do modelo

## Desempenho do Modelo

- **Acurácia de Treino:** 94.76%
- **Acurácia de Teste:** 87.23%
- **Precisão:** 87.82%
- **Recall:** 87.23%
- **Dataset:** 96.088 amostras
- **Features:** 230 sintomas
- **Classes:** 100 diagnósticos

## Como Executar

### Pré-requisitos

- **Python 3.11** (recomendado) ou 3.9 - 3.12
  - ⚠️ **Não use Python 3.14+** (incompatibilidade com algumas dependências)
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/projeto-final-ia.git
cd projeto-final-ia
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. **IMPORTANTE:** Baixe o dataset e treine o modelo:
   
   **Passo 3.1 - Baixar o Dataset:**
   - Acesse: [SymScan Dataset no Kaggle](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)
   - Faça login no Kaggle (crie uma conta se necessário)
   - Clique em **"Download"** (arquivo ZIP ~45 MB)
   - Extraia o arquivo `Diseases_and_Symptoms_dataset.csv`
   - Coloque na pasta `data/` do projeto
   
   **Passo 3.2 - Treinar o Modelo:**
   ```bash
   python train_model_real.py
   ```
   
   Este processo irá:
   - Carregar o dataset (96.088 amostras)
   - Treinar o Random Forest (pode levar alguns minutos)
   - Salvar o modelo treinado em `data/model_real.pkl`
   - Exibir métricas de desempenho

4. Execute a aplicação:
```bash
streamlit run app.py
```
ou
```bash
python -m streamlit run app.py
```

5. Acesse no navegador:
```
http://localhost:8501
```

## Estrutura do Projeto

```
projeto-final-ia/
│
├── app.py                          # Interface Streamlit
├── train_model_real.py             # Script de treinamento do modelo
├── emergency_level.py              # Sistema de níveis de emergência
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação
├── .gitignore                      # Arquivos ignorados pelo Git
│
└── data/
    ├── Diseases_and_Symptoms_dataset.csv  # Dataset principal (baixar do Kaggle)
    └── model_real.pkl                     # Modelo treinado (gerado após treino)
```

> **⚠️ Nota:** Os arquivos `model_real.pkl` e `Diseases_and_Symptoms_dataset.csv` não estão incluídos no repositório devido ao tamanho (>100MB). Você deve baixar o dataset do Kaggle e treinar o modelo localmente.

## Tecnologias Utilizadas

- **Python 3.11** (recomendado)
- **Scikit-learn 1.7.2** - Machine Learning
- **Streamlit 1.28.1** - Interface Web
- **Pandas 2.1.1** - Manipulação de dados
- **NumPy 1.26.4** - Computação numérica
- **Plotly 5.17.0** - Visualizações interativas
- **Matplotlib 3.8.1** - Gráficos estáticos
- **Joblib 1.3.2** - Serialização do modelo

## Metodologia

### Algoritmo: Random Forest Classifier
- **500 árvores de decisão**
- **Profundidade máxima: 50**
- **Estratégia de features:** sqrt
- **Divisão:** 80% treino / 20% teste
- **Balanceamento de classes:** ativado

### Pipeline de Treinamento
1. Carregamento do dataset
2. Pré-processamento e codificação de labels
3. Divisão treino/teste com estratificação
4. Treinamento do Random Forest
5. Avaliação de métricas
6. Serialização do modelo

## Níveis de Emergência

O sistema classifica automaticamente o diagnóstico em 4 níveis:

- 🟢 **Verde (Baixa):** Consultar em dias - posto de saúde
- 🟡 **Amarelo (Urgência):** Consultar em horas - UPA
- 🟠 **Laranja (Emergência):** Procurar pronto-socorro hoje
- 🔴 **Vermelho (Crítica):** Ligar 192 imediatamente

## Aviso Importante

**ESTE SISTEMA É APENAS PARA FINS EDUCACIONAIS**

- ❌ Não substitui consulta médica profissional
- ❌ Não deve ser usado para decisões de tratamento
- ✅ Em caso de emergência, ligue **192** ou procure o pronto-socorro
- ✅ Sempre consulte um médico qualificado

## Testes

Para testar o modelo após o treinamento:

```bash
python train_model_real.py
```

O script irá:
1. Carregar o dataset
2. Treinar o modelo
3. Exibir métricas de desempenho
4. Mostrar os 10 sintomas mais importantes
5. Salvar o modelo treinado

## Melhorias Futuras

- [ ] Adicionar mais datasets médicos
- [ ] Implementar rede neural profunda
- [ ] Integração com APIs de saúde
- [ ] Sistema de histórico de diagnósticos
- [ ] Multilíngue (EN, ES, PT)
- [ ] App mobile (Flutter/React Native)
- [ ] Explicabilidade com SHAP/LIME

## Autor

Desenvolvido como trabalho final do curso de Inteligência Artificial.

## Agradecimentos

- Dataset: [SymScan - Kaggle](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)
- Comunidade Streamlit
- Scikit-learn Documentation

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
