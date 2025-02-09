import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

#load
def load_data(file_path):
    """Load and perform initial data preprocessing"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

class ImprovedPCAImputer:
    def __init__(self, max_iter=100, tol=1e-6, n_components=0.95):
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, X):
        X_filled = X.copy()
        
        # Initial fill with median
        for column in X.columns:
            mask = X[column].isnull()
            X_filled.loc[mask, column] = X[column].median()
        
        # Scale data before PCA
        X_scaled = self.scaler.fit_transform(X_filled)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Iterative PCA imputation
        for iteration in range(self.max_iter):
            X_old = X_scaled.copy()
            
            # Perform PCA with explained variance ratio
            self.pca = PCA(n_components=self.n_components)
            X_pca = self.pca.fit_transform(X_scaled)
            X_reconstructed = self.pca.inverse_transform(X_pca)
            X_reconstructed = pd.DataFrame(X_reconstructed, columns=X.columns)
            
            # Update only missing values
            for column in X.columns:
                mask = X[column].isnull()
                X_scaled.loc[mask, column] = X_reconstructed.loc[mask, column]
            
            # Check convergence
            if np.allclose(X_old, X_scaled, atol=self.tol):
                print(f"PCA converged after {iteration + 1} iterations")
                break
        
        # Inverse transform scaling
        X_final = pd.DataFrame(
            self.scaler.inverse_transform(X_scaled),
            columns=X.columns
        )
        return X_final, self.pca

class DataImputer:
    @staticmethod
    def knn_impute(X, n_neighbors=5, weights='uniform'):
        """Enhanced KNN imputation with weighted neighbors"""
        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric='nan_euclidean'
        )
        return pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        ), imputer
    
    @staticmethod
    def pca_impute(X, max_iter=100, tol=1e-6, n_components=0.95):
        """Improved PCA imputation with variance retention"""
        imputer = ImprovedPCAImputer(
            max_iter=max_iter,
            tol=tol,
            n_components=n_components
        )
        return imputer.fit_transform(X)
    
    @staticmethod
    def mice_impute(X, max_iter=50, random_state=42):
        """Enhanced MICE with Random Forest estimator"""
        rf_estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        )
        
        imputer = IterativeImputer(
            estimator=rf_estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy='median',
            imputation_order='random',
            min_value=X.min().min(),
            max_value=X.max().max(),
            verbose=1
        )
        
        return pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        ), imputer

    @staticmethod
    def xgboost_impute(X, max_iter=10, random_state=42):
        """XGBoost-based iterative imputation"""
        estimator = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy='median',
            imputation_order='random'
        )
        
        return pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        ), imputer

    @staticmethod
    def lightgbm_impute(X, max_iter=10, random_state=42):
        """LightGBM-based iterative imputation"""
        estimator = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            random_state=random_state
        )
        
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy='median',
            imputation_order='random'
        )
        
        return pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        ), imputer

    @staticmethod
    def soft_impute(X, max_iter=100, threshold=1e-5):
        """Soft imputation using matrix completion"""
        from sklearn.impute import SimpleImputer
        
        # Initial fill with mean
        initial_imputer = SimpleImputer(strategy='mean')
        filled_matrix = initial_imputer.fit_transform(X)
        
        mask = np.isnan(X.values)
        prev_matrix = None
        
        for _ in range(max_iter):
            # SVD decomposition
            U, s, Vt = np.linalg.svd(filled_matrix, full_matrices=False)
            
            # Soft thresholding
            s_threshold = np.maximum(s - threshold, 0)
            reconstructed = U @ np.diag(s_threshold) @ Vt
            
            # Update only missing values
            filled_matrix[mask] = reconstructed[mask]
            
            # Check convergence
            if prev_matrix is not None:
                diff = np.abs(filled_matrix - prev_matrix).max()
                if diff < threshold:
                    break
                    
            prev_matrix = filled_matrix.copy()
        
        return pd.DataFrame(filled_matrix, columns=X.columns), None
#preprocess data
def preprocess_data(df, target_col='Biopsy', imputation_method='xgboost', **kwargs):
    """Improved preprocessing with enhanced imputation and SMOTE balancing"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle missing values
    if imputation_method == 'knn':
        X_imputed, imputer = DataImputer.knn_impute(
            X, 
            n_neighbors=kwargs.get('n_neighbors', 5),
            weights=kwargs.get('weights', 'uniform')
        )
    elif imputation_method == 'pca':
        X_imputed, imputer = DataImputer.pca_impute(
            X, 
            max_iter=kwargs.get('max_iter', 100),
            tol=kwargs.get('tol', 1e-6),
            n_components=kwargs.get('n_components', 0.95)
        )
    elif imputation_method == 'mice':
        X_imputed, imputer = DataImputer.mice_impute(
            X,
            max_iter=kwargs.get('max_iter', 50),
            random_state=kwargs.get('random_state', 42)
        )
    elif imputation_method == 'xgboost':
        X_imputed, imputer = DataImputer.xgboost_impute(
            X,
            max_iter=kwargs.get('max_iter', 10),
            random_state=kwargs.get('random_state', 42)
        )
    elif imputation_method == 'lightgbm':
        X_imputed, imputer = DataImputer.lightgbm_impute(
            X,
            max_iter=kwargs.get('max_iter', 10),
            random_state=kwargs.get('random_state', 42)
        )
    elif imputation_method == 'soft':
        X_imputed, imputer = DataImputer.soft_impute(
            X,
            max_iter=kwargs.get('max_iter', 100),
            threshold=kwargs.get('threshold', 1e-5)
        )
    else:
        raise ValueError(f"Unsupported imputation method: {imputation_method}")
    
    # Scale features after imputation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply SMOTE after scaling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=kwargs.get('random_state', 42))
    X_balanced, y_balanced = smote.fit_resample(X_scaled_df, y)
    
    return X_balanced, y_balanced, scaler, imputer
#data cleaning
def split_data(X, y, train_percentage, random_state=42):
    """Split data into training and testing sets"""
    if not 0 < train_percentage < 1:
        raise ValueError("train_percentage must be between 0 and 1")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=1-train_percentage,
        random_state=random_state,
        stratify=y
    )
    
    print("\nData Split Information:")
    print(f"Training set: {len(X_train)} samples ({train_percentage*100:.1f}%)")
    print(f"Testing set:  {len(X_test)} samples ({(1-train_percentage)*100:.1f}%)")
    
    print("\nClass Distribution:")
    print("Training set:", pd.Series(y_train).value_counts(normalize=True).mul(100).round(1).to_dict())
    print("Testing set:", pd.Series(y_test).value_counts(normalize=True).mul(100).round(1).to_dict())
    
    return X_train, X_test, y_train, y_test

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
        
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

def evaluate_imputation(original_df, imputed_df, target_col='Biopsy'):
    """Evaluate imputation quality"""
    original = original_df.drop(target_col, axis=1)
    imputed = imputed_df.drop(target_col, axis=1)
    
    missing_mask = original.isnull()
    
    metrics = {
        'total_missing': missing_mask.sum().sum(),
        'missing_percentage': (missing_mask.sum().sum() / (original.shape[0] * original.shape[1]) * 100),
        'imputed_summary': imputed[missing_mask].describe() if missing_mask.sum().sum() > 0 else pd.DataFrame(),
        'original_summary': original[~missing_mask].describe()
    }
    
    return metrics