# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'download_over_limit']
        self.base_numeric_columns = ['subscription_age', 'bill_avg', 'reamining_contract', 
                                    'service_failure_count', 'download_avg', 'upload_avg']
        self.numeric_columns = self.base_numeric_columns.copy()
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path, na_values=['', ' ', 'NA', 'null', 'NULL'])
        print(f"Завантажено {len(df)} рядків")
        return df
    
    def clean_data(self, df):
        df_clean = df.copy()
        
        if 'id' in df_clean.columns:
            df_clean = df_clean.drop('id', axis=1)
        
        print("\nПеревірка пропусків до обробки:")
        missing_before = df_clean.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Обробка reamining_contract
        if 'reamining_contract' in df_clean.columns:
            df_clean['reamining_contract_missing'] = df_clean['reamining_contract'].isnull().astype(int)
            print("Створено індикатор пропусків для 'reamining_contract'")
            
            median_val = df_clean['reamining_contract'].median()
            df_clean['reamining_contract'].fillna(median_val, inplace=True)
            print(f"Заповнено пропуски в 'reamining_contract' медіаною: {median_val:.2f}")
            
            if 'reamining_contract_missing' not in self.numeric_columns:
                self.numeric_columns.append('reamining_contract_missing')
        
        # Заповнення пропусків для всіх числових колонок
        for col in self.base_numeric_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"Заповнено пропуски в '{col}' медіаною: {median_val:.2f}")
        
        # Для категоріальних колонок
        for col in self.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"Заповнено пропуски в '{col}' модою: {mode_val}")
        
        # ВАЖЛИВО: Переконуємося, що NaN більше немає
        # Заповнюємо всі залишки NaN нулем (безпечно)
        if df_clean.isnull().any().any():
            print("\nЗаповнення залишкових NaN значенням 0...")
            df_clean = df_clean.fillna(0)
        
        missing_after = df_clean.isnull().sum()
        if missing_after.sum() > 0:
            print("\nЗалишились пропуски:")
            print(missing_after[missing_after > 0])
        else:
            print("\nВсі пропуски успішно оброблено!")
        
        return df_clean
    
    def encode_categorical(self, df):
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Закодовано колонку: {col}")
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        df_scaled = df.copy()
        
        available_numeric = [col for col in self.numeric_columns if col in df_scaled.columns]
        print(f"Колонки для нормалізації: {available_numeric}")
        
        if fit:
            print(f"Навчання StandardScaler на {len(available_numeric)} ознаках")
            self.scaler.fit(df_scaled[available_numeric])
            scaled_data = self.scaler.transform(df_scaled[available_numeric])
        else:
            scaled_data = self.scaler.transform(df_scaled[available_numeric])
        
        for i, col in enumerate(available_numeric):
            df_scaled[col] = scaled_data[:, i]
        
        print("Нормалізацію завершено")
        return df_scaled
    
    def prepare_features(self, df, is_training=True):
        print("\n" + "="*50)
        print("ПОЧАТОК ПІДГОТОВКИ ДАНИХ")
        print("="*50)
        
        df_clean = self.clean_data(df)
        
        target = None
        if 'churn' in df_clean.columns:
            target = df_clean['churn']
            if not is_training:
                df_clean = df_clean.drop('churn', axis=1)
                print("Колонка 'churn' видалена для тестових даних")
            else:
                df_clean = df_clean.drop('churn', axis=1)
        
        df_encoded = self.encode_categorical(df_clean)
        
        self.feature_columns = df_encoded.columns.tolist()
        print(f"\nОзнаки для моделі ({len(self.feature_columns)}): {self.feature_columns}")
        
        df_scaled = self.scale_features(df_encoded, fit=is_training)
        
        if is_training and target is not None:
            print("\nПідготовку даних завершено")
            print(f"\nРозподіл цільової змінної:")
            print(f"  Відтік (1): {target.sum()} ({target.sum()/len(target)*100:.1f}%)")
            print(f"  Залишились (0): {len(target)-target.sum()} ({(len(target)-target.sum())/len(target)*100:.1f}%)")
            return df_scaled, target
        
        print("\nПідготовку даних завершено")
        return df_scaled
    
    def split_data(self, features, target, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state, 
            stratify=target
        )
        print(f"\nРозподіл даних:")
        print(f"  Тренувальний набір: {len(X_train)} зразків")
        print(f"  Тестовий набір: {len(X_test)} зразків")
        print(f"\nРозподіл цільової змінної:")
        print(f"  Тренування - Відтік: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Тест - Відтік: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'base_numeric_columns': self.base_numeric_columns
        }
        joblib.dump(preprocessor_data, path)
        print(f"\nПрепроцесор збережено в {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        self.numeric_columns = data['numeric_columns']
        self.categorical_columns = data['categorical_columns']
        if 'base_numeric_columns' in data:
            self.base_numeric_columns = data['base_numeric_columns']
        print(f"Препроцесор завантажено з {path}")

if __name__ == "__main__":
    print("ЗАПУСК ТЕСТУВАННЯ ПРЕПРОЦЕСОРА")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/raw/internet_service_churn.csv')
    features, target = preprocessor.prepare_features(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)
    preprocessor.save_preprocessor()
    
    print("\n" + "="*50)
    print("ТЕСТ ПРЕПРОЦЕСОРА УСПІШНО ЗАВЕРШЕНО!")
    print(f"Форма ознак: {features.shape}")
    print(f"Кількість ознак: {len(preprocessor.feature_columns)}")