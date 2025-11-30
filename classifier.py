import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class DiagnosticClassifier:
    """
    Classificador para diagnóstico baseado em sintomas.
    Utiliza Random Forest para classificação multi-classe.
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptom_encoder = None
        self.symptoms_list = None
        self.diagnoses = None
        
    def create_dataset(self):
        """
        Cria um dataset sintético com sintomas e diagnósticos.
        """
        # Lista de sintomas possíveis
        self.symptoms_list = [
            'Febre', 'Tosse', 'Espirro', 'Dor de garganta',
            'Dor de cabeça', 'Falta de ar', 'Dor no peito',
            'Náusea', 'Vômito', 'Diarréia', 'Dor abdominal',
            'Fadiga', 'Calafrios', 'Congestão nasal'
        ]
        
        # Dataset: [sintomas...] -> diagnóstico
        data = []
        diagnoses_dict = {
            'Gripe': {'sintomas': ['Febre', 'Tosse', 'Dor de cabeça', 'Fadiga', 'Calafrios'], 'samples': 50},
            'COVID-19': {'sintomas': ['Febre', 'Tosse', 'Falta de ar', 'Fadiga', 'Dor de cabeça'], 'samples': 50},
            'Pneumonia': {'sintomas': ['Febre', 'Tosse', 'Dor no peito', 'Falta de ar'], 'samples': 40},
            'Bronquite': {'sintomas': ['Tosse', 'Falta de ar', 'Fadiga', 'Congestão nasal'], 'samples': 35},
            'Apendicite': {'sintomas': ['Dor abdominal', 'Náusea', 'Vômito', 'Febre'], 'samples': 30},
            'Gastroenterite': {'sintomas': ['Diarréia', 'Náusea', 'Vômito', 'Dor abdominal'], 'samples': 45},
            'Resfriado': {'sintomas': ['Espirro', 'Congestão nasal', 'Dor de garganta', 'Tosse'], 'samples': 55},
            'Enxaqueca': {'sintomas': ['Dor de cabeça', 'Náusea', 'Fadiga'], 'samples': 40},
        }
        
        self.diagnoses = list(diagnoses_dict.keys())
        
        # Gerar amostras
        for diagnosis, info in diagnoses_dict.items():
            sintomas_principais = set(info['sintomas'])
            
            for _ in range(info['samples']):
                # Criar amostra com sintomas principais + alguns aleatórios
                amostra = {}
                
                # Sintomas principais com alta probabilidade
                for sintoma in self.symptoms_list:
                    if sintoma in sintomas_principais:
                        amostra[sintoma] = np.random.choice([0, 1], p=[0.2, 0.8])
                    else:
                        amostra[sintoma] = np.random.choice([0, 1], p=[0.85, 0.15])
                
                amostra['Diagnóstico'] = diagnosis
                data.append(amostra)
        
        df = pd.DataFrame(data)
        return df
    
    def train(self, df):
        """
        Treina o modelo de classificação.
        """
        X = df[self.symptoms_list].values
        y = df['Diagnóstico'].values
        
        # Encoder para diagnósticos
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Treinar Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Métricas
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, target_names=self.diagnoses),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(self.symptoms_list, self.model.feature_importances_))
        }
        
        return metrics
    
    def predict(self, symptoms_dict):
        """
        Realiza predição dado um dicionário de sintomas.
        
        Args:
            symptoms_dict: {'Febre': 1, 'Tosse': 0, ...}
        
        Returns:
            diagnosis: diagnóstico previsto
            probability: probabilidade da predição
            all_probabilities: probabilidades para todas as classes
        """
        X = [symptoms_dict[s] for s in self.symptoms_list]
        X = np.array(X).reshape(1, -1)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        diagnosis = self.label_encoder.inverse_transform([prediction])[0]
        probability = max(probabilities)
        
        all_probs = dict(zip(self.diagnoses, probabilities))
        
        return diagnosis, probability, all_probs
    
    def get_feature_importance(self):
        """Retorna importância das features."""
        return dict(zip(self.symptoms_list, self.model.feature_importances_))
    
    def save(self, path='model.pkl'):
        """Salva o modelo treinado."""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path='model.pkl'):
        """Carrega modelo treinado."""
        return joblib.load(path)


# Script para treinar e salvar modelo
if __name__ == '__main__':
    print("Criando e treinando modelo de diagnóstico...")
    
    classifier = DiagnosticClassifier()
    df = classifier.create_dataset()
    
    # Salvar dataset
    df.to_csv('data/symptoms_data.csv', index=False)
    print(f"✓ Dataset criado: {len(df)} amostras")
    
    # Treinar
    metrics = classifier.train(df)
    print(f"✓ Modelo treinado com acurácia: {metrics['accuracy']:.2%}")
    
    # Salvar modelo
    os.makedirs('data', exist_ok=True)
    classifier.save('data/model.pkl')
    print("✓ Modelo salvo: data/model.pkl")