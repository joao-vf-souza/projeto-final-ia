"""
train_model_real.py

Script de treinamento do modelo de diagnóstico médico usando Random Forest.
Utiliza o dataset SymScan do Kaggle com 96.088 amostras, 230 sintomas e 100 doenças.

Uso:
    python train_model_real.py

Pré-requisitos:
    - Dataset 'Diseases_and_Symptoms_dataset.csv' na pasta data/
    - Dependências instaladas (requirements.txt)

Saída:
    - Modelo treinado salvo em data/model_real.pkl
    - Métricas de desempenho exibidas no console
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

class DiagnosticClassifierReal:
    """
    Classificador de diagnósticos médicos usando Random Forest.
    
    Atributos:
        model: Modelo Random Forest treinado
        label_encoder: Codificador de labels para diagnósticos
        symptoms_list: Lista de sintomas utilizados como features
        diagnoses: Lista de diagnósticos possíveis
        feature_importance: Importância de cada sintoma na predição
    """
    
    def __init__(self):
        """Inicializa o classificador com atributos vazios."""
        self.model = None
        self.label_encoder = None
        self.symptoms_list = None
        self.diagnoses = None
        self.feature_importance = None
        self.metrics = None  # Armazena as métricas de treinamento
    
    def load_real_dataset(self, csv_path):
        """
        Carrega e processa o dataset SymScan.
        
        Formato esperado do CSV:
            - Primeira coluna: nome da doença/diagnóstico
            - Demais colunas: sintomas binários (0 = ausente, 1 = presente)
        
        Parâmetros:
            csv_path (str): Caminho para o arquivo CSV do dataset
        
        Retorna:
            DataFrame: Dataset carregado e processado
        """
        print(f"Carregando dataset de: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"   Shape original: {df.shape}")
        
        # Identifica colunas: primeira é diagnóstico, restantes são sintomas
        disease_col = df.columns[0]
        symptom_cols = df.columns[1:].tolist()
        
        # Armazena diagnósticos únicos e lista de sintomas
        self.diagnoses = df[disease_col].unique().tolist()
        self.symptoms_list = symptom_cols
        
        print(f"Dataset carregado com sucesso!")
        print(f"   - Amostras: {len(df)}")
        print(f"   - Sintomas: {len(self.symptoms_list)}")
        print(f"   - Doenças: {len(self.diagnoses)}")
        
        return df
    
    def train(self, df):
        """
        Treina modelo Random Forest com dataset real
        """
        print("\nIniciando treinamento...")
        
        X = df[self.symptoms_list].values
        y = df.iloc[:, 0].values  # Primeira coluna é o diagnóstico
        
        # Split dos dados
        print("   - Dividindo dados: 80% treino, 20% teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Encode labels
        print("   - Codificando diagnósticos...")
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Treinar modelo com hiperparâmetros otimizados para melhor generalização
        print("   - Treinando Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=300,           # Reduzir árvores para evitar overfitting
            max_depth=40,               # Limitar profundidade para melhor generalização
            min_samples_split=5,        # Aumentar para 5 - evita splits muito específicos
            min_samples_leaf=2,         # Aumentar para 2 - folhas mais robustas
            max_features='log2',        # log2 geralmente melhor que sqrt em alta dimensão
            min_impurity_decrease=0.0001,  # Pequena penalidade para splits
            max_samples=0.8,            # Usar 80% dos dados por árvore (bagging mais forte)
            bootstrap=True,             # Manter bootstrap ativado
            n_jobs=-1,
            random_state=42,
            verbose=0,
            criterion='gini',
            class_weight='balanced',
            ccp_alpha=0.001             # Pruning para reduzir overfitting
        )
        
        self.model.fit(X_train, y_train_encoded)
        
        # Prever e calcular métricas principais
        print("\nCalculando métricas finais...")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        acc_train = accuracy_score(y_train_encoded, y_pred_train)
        acc_test = accuracy_score(y_test_encoded, y_pred_test)
        precision = precision_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred_test, average='weighted', zero_division=0)
        
        # Feature importance
        self.feature_importance = dict(
            zip(self.symptoms_list, self.model.feature_importances_)
        )
        
        self.metrics = {
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'precision': precision,
            'recall': recall,
            'n_samples': len(df),
            'n_symptoms': len(self.symptoms_list),
            'n_diseases': len(self.diagnoses)
        }
        
        return self.metrics
    
    def get_feature_importance(self):
        """Retorna a importância de cada feature/sintoma"""
        if self.feature_importance is None:
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(
                    zip(self.symptoms_list, self.model.feature_importances_)
                )
            else:
                return {}
        return self.feature_importance
    
    def predict(self, symptoms_dict):
        """
        Realiza predição baseada nos sintomas fornecidos
        
        Args:
            symptoms_dict: dicionário {sintoma: valor_binário}
        
        Returns:
            tuple: (diagnóstico, confiança, todas_probabilidades)
        """
        # Criar vetor de features
        X = np.array([[symptoms_dict.get(s, 0) for s in self.symptoms_list]])
        
        # Predição
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0]
        
        # Decodificar diagnóstico
        diagnosis = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = y_proba[y_pred]
        
        # Todas as probabilidades
        all_probabilities = dict(zip(
            self.label_encoder.classes_,
            y_proba
        ))
        
        return diagnosis, confidence, all_probabilities
    
    def save(self, path='data/model_real.pkl'):
        """Salva o modelo treinado"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)
        print(f"✓ Modelo salvo em: {path}")


def main():
    print("=" * 80)
    print("TREINANDO MODELO COM DATASET REAL SYMPSCAN")
    print("=" * 80)
    
    # Verificar se arquivo existe
    dataset_path = 'data/Diseases_and_Symptoms_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\nERRO: Dataset não encontrado em {dataset_path}")
        print("\nO que fazer:")
        print("1. Acesse: https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease")
        print("2. Clique em 'Download'")
        print("3. Descompacte o arquivo")
        print("4. Copie 'Diseases_and_Symptoms_dataset.csv' para a pasta 'data/'")
        print("5. Execute novamente: python train_model_real.py")
        return
    
    try:
        # Criar classificador
        classifier = DiagnosticClassifierReal()
        
        # Carregar dataset
        df = classifier.load_real_dataset(dataset_path)
        
        # Treinar modelo
        metrics = classifier.train(df)
        
        # Exibir resultados
        print("\n" + "=" * 80)
        print("RESULTADOS DO TREINAMENTO")
        print("=" * 80)
        print(f"\nMétricas:")
        print(f"   Acurácia (Treino):     {metrics['accuracy_train']:.2%}")
        print(f"   Acurácia (Teste):      {metrics['accuracy_test']:.2%}")
        print(f"   Precisão:              {metrics['precision']:.2%}")
        print(f"   Recall:                {metrics['recall']:.2%}")
        
        # Análise de generalização
        gap = metrics['accuracy_train'] - metrics['accuracy_test']
        print(f"\n   Gap Treino-Teste:      {gap:.2%}", end="")
        if gap < 0.05:
            print(" (Excelente generalização)")
        elif gap < 0.10:
            print(" (Boa generalização)")
        else:
            print(" (Overfitting detectado)")
        
        print(f"\nDataset:")
        print(f"   Amostras:          {metrics['n_samples']:,}")
        print(f"   Sintomas:          {metrics['n_symptoms']}")
        print(f"   Doenças:           {metrics['n_diseases']}")
        
        print(f"\nTop 10 Sintomas Mais Importantes:")
        top_features = sorted(
            classifier.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (symptom, importance) in enumerate(top_features, 1):
            bar = "█" * int(importance * 100)
            print(f"   {i:2d}. {symptom:30s} {bar} {importance:.1%}")
        
        # Salvar modelo
        print("\nSalvando modelo...")
        classifier.save('data/model_real.pkl')
        
        # Informações finais
        print("\n" + "=" * 80)
        print("PRÓXIMOS PASSOS")
        print("=" * 80)
        print("\n1. Execute o Streamlit:")
        print("   python -m streamlit run app.py")
        print("\n2. Abra no navegador:")
        print("   http://localhost:8501")
        print("\n3. Teste o sistema com o modelo treinado!")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERRO durante treinamento:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nVerifique se:")
        print("  - O arquivo CSV está no caminho correto")
        print("  - O arquivo não está corrompido")
        print("  - Você tem espaço em disco suficiente")
        raise


if __name__ == '__main__':
    main()