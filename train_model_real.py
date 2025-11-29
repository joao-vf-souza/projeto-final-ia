"""
train_model_real.py
Script para treinar modelo com dataset real SympScan
Execute ANTES de rodar o Streamlit: python train_model_real.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import joblib

class DiagnosticClassifierReal:
    """
    Classificador para dataset real SympScan
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptoms_list = None
        self.diagnoses = None
        self.feature_importance = None
    
    def load_real_dataset(self, csv_path):
        """
        Carrega dataset real do SympScan
        
        Formato esperado:
        - Primeira coluna: doença/diagnóstico
        - Demais colunas: sintomas (0 ou 1)
        """
        print(f"📂 Carregando dataset de: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"   Shape original: {df.shape}")
        
        # Primeira coluna é o diagnóstico
        disease_col = df.columns[0]
        symptom_cols = df.columns[1:].tolist()
        
        self.diagnoses = df[disease_col].unique().tolist()
        self.symptoms_list = symptom_cols
        
        print(f"✓ Dataset carregado!")
        print(f"   - Amostras: {len(df)}")
        print(f"   - Sintomas: {len(self.symptoms_list)}")
        print(f"   - Doenças: {len(self.diagnoses)}")
        
        return df
    
    def train(self, df):
        """
        Treina modelo Random Forest com dataset real
        """
        print("\n🤖 Iniciando treinamento...")
        
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
        
        # Treinar modelo
        print("   - Treinando Random Forest (200 árvores)...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        self.model.fit(X_train, y_train_encoded)
        
        # Prever e calcular métricas
        print("\n📊 Calculando métricas...")
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
        
        metrics = {
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'precision': precision,
            'recall': recall,
            'n_samples': len(df),
            'n_symptoms': len(self.symptoms_list),
            'n_diseases': len(self.diagnoses)
        }
        
        return metrics
    
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
    print("🏥 TREINANDO MODELO COM DATASET REAL SYMPSCAN")
    print("=" * 80)
    
    # Verificar se arquivo existe
    dataset_path = 'data/Diseases_and_Symptoms_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERRO: Dataset não encontrado em {dataset_path}")
        print("\n📋 O que fazer:")
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
        print("✅ RESULTADOS DO TREINAMENTO")
        print("=" * 80)
        print(f"\n📊 Métricas:")
        print(f"   Acurácia (Treino): {metrics['accuracy_train']:.2%}")
        print(f"   Acurácia (Teste):  {metrics['accuracy_test']:.2%}")
        print(f"   Precisão:          {metrics['precision']:.2%}")
        print(f"   Recall:            {metrics['recall']:.2%}")
        
        print(f"\n📈 Dataset:")
        print(f"   Amostras:          {metrics['n_samples']:,}")
        print(f"   Sintomas:          {metrics['n_symptoms']}")
        print(f"   Doenças:           {metrics['n_diseases']}")
        
        print(f"\n🎯 Top 10 Sintomas Mais Importantes:")
        top_features = sorted(
            classifier.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (symptom, importance) in enumerate(top_features, 1):
            bar = "█" * int(importance * 100)
            print(f"   {i:2d}. {symptom:30s} {bar} {importance:.1%}")
        
        # Salvar modelo
        print("\n💾 Salvando modelo...")
        classifier.save('data/model_real.pkl')
        
        # Informações finais
        print("\n" + "=" * 80)
        print("🚀 PRÓXIMOS PASSOS")
        print("=" * 80)
        print("\n1. Execute o Streamlit:")
        print("   streamlit run app.py")
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