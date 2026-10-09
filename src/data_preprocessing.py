# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'download_over_limit']
        self.base_numeric_columns = ['subscription_age', 'bill_avg', 'remaining_contract', 
                                    'service_failure_count', 'download_avg', 'upload_avg']
        self.numeric_columns = self.base_numeric_columns.copy()
        
    def load_data(self, file_path):
        """Loading data from a CSV file"""
        df = pd.read_csv(file_path, na_values=['', ' ', 'NA', 'null', 'NULL'])
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
        return df
    
    def clean_data(self, df):
        """Cleaning data from missing values and anomalies"""
        df_clean = df.copy()
        
        if 'id' in df_clean.columns:
            df_clean = df_clean.drop('id', axis=1)
        
        print("\nChecking for missing values before processing:")
        missing_before = df_clean.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Обробка remaining_contract
        if 'remaining_contract' in df_clean.columns:
            df_clean['remaining_contract_missing'] = df_clean['remaining_contract'].isnull().astype(int)
            print("Created missing value indicator for 'remaining_contract'")
            
            median_val = df_clean['remaining_contract'].median()
            df_clean['remaining_contract'] = df_clean['remaining_contract'].fillna(median_val)
            print(f"Filled missing values in 'remaining_contract' with median: {median_val:.2f}")
            
            if 'remaining_contract_missing' not in self.numeric_columns:
                self.numeric_columns.append('remaining_contract_missing')
        
        for col in self.base_numeric_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val:.2f}")
        
        for col in self.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
        
        if df_clean.isnull().any().any():
            print("\nFilling remaining NaN values with 0...")
            df_clean = df_clean.fillna(0)
        
        missing_after = df_clean.isnull().sum()
        if missing_after.sum() > 0:
            print("\nRemaining missing values:")
            print(missing_after[missing_after > 0])
        else:
            print("\nAll missing values successfully processed!")
        
        return df_clean
    
    def encode_categorical(self, df):
        """Encoding categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """Scaling numeric features using the already trained scaler"""
        df_scaled = df.copy()
        
        available_numeric = [col for col in self.numeric_columns if col in df_scaled.columns]
        if not available_numeric:
            print("No numeric columns available for scaling")
            return df_scaled
        
        print(f"Columns for normalization: {available_numeric}")
        
        if fit:
            print(f"Training StandardScaler on {len(available_numeric)} features")
            self.scaler.fit(df_scaled[available_numeric])
            scaled_data = self.scaler.transform(df_scaled[available_numeric])
        else:
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError("Scaler not trained. Please train it on the training data first.")
            scaled_data = self.scaler.transform(df_scaled[available_numeric])
        
        for i, col in enumerate(available_numeric):
            df_scaled[col] = scaled_data[:, i]
        
        print("Normalization completed")
        return df_scaled
    
    def prepare_features(self, df, is_training=True):
        """Preparation of all features for the model"""
        print("\n" + "="*50)
        print("STARTING DATA PREPARATION")
        print("="*50)
        
        df_clean = self.clean_data(df)
        
        target = None
        if 'churn' in df_clean.columns:
            target = df_clean['churn']
            if not is_training:
                df_clean = df_clean.drop('churn', axis=1)
                print("Column 'churn' removed for test data")
            else:
                df_clean = df_clean.drop('churn', axis=1)
        
        df_encoded = self.encode_categorical(df_clean)
        
        if self.feature_columns is None:
            self.feature_columns = df_encoded.columns.tolist()
        else:
            for col in self.feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[self.feature_columns]
        
        print(f"\nFeatures for the model ({len(self.feature_columns)}): {self.feature_columns}")
        
        if is_training:
            self.feature_columns = df_encoded.columns.tolist()
            if target is not None:
                self.save_processed_data(df_encoded, target)
                print("\nData preparation completed")
                print(f"\nTarget variable distribution:")
                print(f"  Churn (1): {target.sum()} ({target.sum()/len(target)*100:.1f}%)")
                print(f"  Not Churn (0): {len(target)-target.sum()} ({(len(target)-target.sum())/len(target)*100:.1f}%)")
                return df_encoded, target
        else:
            df_scaled = self.scale_features(df_encoded, fit=False)
            print("\nData preparation completed")
            return df_scaled
        
        print("\nData preparation completed")
        return df_encoded
    
    def save_processed_data(self, features, target):
        """Saving processed data in data/processed/"""
        os.makedirs('data/processed', exist_ok=True)
        
        # Combine features and target
        processed_df = features.copy()
        processed_df['churn'] = target.values
        
        # Save the processed data
        processed_df.to_csv('data/processed/processed_data.csv', index=False)
        print(f"\nProcessed data saved in data/processed/processed_data.csv")
        print(f"   Size: {processed_df.shape}")
    
    def split_data(self, features, target, test_size=0.2, random_state=42):
        """Splitting data into training and test sets and scaling after splitting"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state, 
            stratify=target
        )
        print(f"\nData split:")

        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"\nTarget variable distribution:")
        print(f"  Training - Churn: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Test - Churn: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")

        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Saving preprocessor"""
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
        print(f"\nPreprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Loading preprocessor"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        self.numeric_columns = data['numeric_columns']
        self.categorical_columns = data['categorical_columns']
        if 'base_numeric_columns' in data:
            self.base_numeric_columns = data['base_numeric_columns']
        print(f"Preprocessor loaded from {path}")

if __name__ == "__main__":
    print("Launching Preprocessor Testing")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/raw/internet_service_churn.csv')
    features, target = preprocessor.prepare_features(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(features, target)
    preprocessor.save_preprocessor()
    
    print("\n" + "="*50)
    print("Preprocessor Testing Completed Successfully!")
    print(f"Feature shape: {features.shape}")
    print(f"Number of features: {len(preprocessor.feature_columns)}")