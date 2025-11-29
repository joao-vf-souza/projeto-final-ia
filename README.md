# 🏥 Sistema de Diagnóstico Médico com IA

Sistema inteligente de diagnóstico baseado em sintomas usando Machine Learning. Desenvolvido como trabalho final do curso de Inteligência Artificial.

## 📋 Sobre o Projeto

Este sistema utiliza **Random Forest Classifier** para prever possíveis diagnósticos médicos baseado em sintomas informados pelo usuário. O modelo foi treinado com o dataset **SymScan** do Kaggle, contendo 96.088 amostras com 230 sintomas diferentes e 100 doenças.

### 🎯 Funcionalidades

- ✅ Diagnóstico baseado em sintomas selecionados
- ✅ Classificação de nível de emergência (Verde, Amarelo, Laranja, Vermelho)
- ✅ Visualização de confiança e probabilidades
- ✅ Interface web interativa com Streamlit
- ✅ Análise de importância de sintomas
- ✅ Métricas e gráficos de desempenho do modelo

## 📊 Desempenho do Modelo

- **Acurácia de Treino:** 92.69%
- **Acurácia de Teste:** 86.34%
- **Precisão:** 87.70%
- **Recall:** 86.34%
- **Dataset:** 96.088 amostras
- **Features:** 230 sintomas
- **Classes:** 100 diagnósticos

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
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

3. Baixe o dataset (se necessário):
   - Acesse: [SymScan Dataset no Kaggle](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)
   - Baixe o arquivo `Diseases_and_Symptoms_dataset.csv`
   - Coloque na pasta `data/`

4. Treine o modelo:
```bash
python train_model_real.py
```

5. Execute a aplicação:
```bash
streamlit run app.py
```
ou
```bash
python -m streamlit run app.py
```

6. Acesse no navegador:
```
http://localhost:8501
```

## 📁 Estrutura do Projeto

```
projeto-final-ia/
│
├── app.py                          # Interface Streamlit
├── train_model_real.py             # Script de treinamento do modelo
├── classifier.py                   # Classe do classificador (demo)
├── emergency_level.py              # Sistema de níveis de emergência
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação
│
└── data/
    ├── Diseases_and_Symptoms_dataset.csv  # Dataset principal
    ├── model_real.pkl                     # Modelo treinado
    ├── description.csv                    # Descrições de doenças
    ├── medications.csv                    # Medicamentos
    ├── precautions.csv                    # Precauções
    ├── diets.csv                          # Dietas recomendadas
    └── workout.csv                        # Exercícios recomendados
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.13**
- **Scikit-learn** - Machine Learning
- **Streamlit** - Interface Web
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Plotly** - Visualizações interativas
- **Matplotlib** - Gráficos estáticos
- **Joblib** - Serialização do modelo

## 🔬 Metodologia

### Algoritmo: Random Forest Classifier
- **200 árvores de decisão**
- **Profundidade máxima: 30**
- **Estratégia de features:** sqrt
- **Divisão:** 80% treino / 20% teste

### Pipeline de Treinamento
1. Carregamento do dataset
2. Pré-processamento e codificação de labels
3. Divisão treino/teste com estratificação
4. Treinamento do Random Forest
5. Avaliação de métricas
6. Serialização do modelo

## 📈 Níveis de Emergência

O sistema classifica automaticamente o diagnóstico em 4 níveis:

- 🟢 **Verde (Baixa):** Consultar em dias - posto de saúde
- 🟡 **Amarelo (Urgência):** Consultar em horas - UPA
- 🟠 **Laranja (Emergência):** Procurar pronto-socorro hoje
- 🔴 **Vermelho (Crítica):** Ligar 192 imediatamente

## ⚠️ Aviso Importante

**ESTE SISTEMA É APENAS PARA FINS EDUCACIONAIS**

- ❌ Não substitui consulta médica profissional
- ❌ Não deve ser usado para decisões de tratamento
- ✅ Em caso de emergência, ligue **192** ou procure o pronto-socorro
- ✅ Sempre consulte um médico qualificado

## 🧪 Testes

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

## 📝 Melhorias Futuras

- [ ] Adicionar mais datasets médicos
- [ ] Implementar rede neural profunda
- [ ] Integração com APIs de saúde
- [ ] Sistema de histórico de diagnósticos
- [ ] Multilíngue (EN, ES, PT)
- [ ] App mobile (Flutter/React Native)
- [ ] Explicabilidade com SHAP/LIME

## 👨‍💻 Autor

Desenvolvido como trabalho final do curso de Inteligência Artificial.

## 📄 Licença

Este projeto é para fins educacionais. Consulte o arquivo LICENSE para mais detalhes.

## 🙏 Agradecimentos

- Dataset: [SymScan - Kaggle](https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease)
- Comunidade Streamlit
- Scikit-learn Documentation

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
