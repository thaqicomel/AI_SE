import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.model_selection import KFold, cross_val_score
#modelfactory
class ModelFactory:
    @staticmethod
    def get_base_models():
        """Get available base models for stack ensemble"""
        return {
            'LR': ('Logistic Regression', LogisticRegression()),
            'RF': ('Random Forest', RandomForestClassifier()),
            'XGB': ('XGBoost', xgb.XGBClassifier()),
            'CAT': ('CatBoost', CatBoostClassifier(verbose=False)),
            'ET': ('Extra Trees', ExtraTreesClassifier()),
            'GB': ('Gradient Boosting', GradientBoostingClassifier())
        }

    @staticmethod
    def get_meta_learners():
        """Get available meta-learners for stack ensemble"""
        return {
            'LR': ('Logistic Regression', LogisticRegression()),
            'RF': ('Random Forest', RandomForestClassifier(n_estimators=100)),
            'XGB': ('XGBoost', xgb.XGBClassifier(n_estimators=100))
        }

    @staticmethod
    def get_base_models():
        """Get available base models for stack ensemble"""
        return {
            'RF': ('Random Forest', RandomForestClassifier(n_estimators=100)),
            'GB': ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100)),
            'ET': ('Extra Trees', ExtraTreesClassifier(n_estimators=100)),
            'LR': ('Logistic Regression', LogisticRegression(max_iter=1000))
        }

    @staticmethod
    def get_meta_learners():
        """Get available meta-learners for stack ensemble"""
        return {
            'LR': ('Logistic Regression', LogisticRegression(max_iter=1000)),
            'RF': ('Random Forest', RandomForestClassifier(n_estimators=100))
        }

    @staticmethod
    def get_model(model_name, random_state=42, stack_config=None):
        # Original single models dictionary - keep as is for individual models
        base_models = {
            'LR': LogisticRegression(
                random_state=random_state,
                max_iter=2000,
                class_weight='balanced',
                C=0.1,
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5
            ),
            'DT': DecisionTreeClassifier(
                random_state=random_state,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3
            ),
            'RF': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            ),
            'XGB': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                scale_pos_weight=14.5,
                min_child_weight=1,
                gamma=0.1
            ),
            'SVC': SVC(
                probability=True,
                random_state=random_state,
                class_weight='balanced',
                C=2.0,
                kernel='rbf',
                gamma='scale'
            ),
            'ET': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            ),
            'CAT': CatBoostClassifier(
                iterations=200,
                learning_rate=0.01,
                depth=6,
                random_state=random_state,
                auto_class_weights='Balanced',
                l2_leaf_reg=3,
                border_count=128,
                verbose=False
            ),
            'ADA': AdaBoostClassifier(random_state=random_state),
            'SGD': SGDClassifier(
                loss='modified_huber',
                max_iter=2000,
                tol=1e-4,
                random_state=random_state,
                class_weight='balanced',
                alpha=0.00005,
                penalty='elasticnet',
                l1_ratio=0.5,
                learning_rate='adaptive',
                eta0=0.005
            ),
            'GB': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=random_state
            )
        }

        # Handle stack ensemble
        if model_name == 'STACK' and stack_config is not None:
            # Create stack-compatible base models
            estimators = []
            for name in stack_config['base_models']:
                if name == 'RF':
                    model = RandomForestClassifier(
                        n_estimators=100, 
                        random_state=random_state,
                        class_weight='balanced'
                    )
                elif name == 'GB':
                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        random_state=random_state
                    )
                elif name == 'ET':
                    model = ExtraTreesClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        class_weight='balanced'
                    )
                elif name == 'LR':
                    model = LogisticRegression(
                        random_state=random_state,
                        max_iter=1000,
                        class_weight='balanced'
                    )
                estimators.append((name, model))

            # Set meta-learner
            meta_learner = stack_config['meta_learner']
            if meta_learner == 'LR':
                final_estimator = LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
            else:  # Default to RF
                final_estimator = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    class_weight='balanced'
                )

            # Create and return stack ensemble
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=stack_config['cv_folds'],
                stack_method='predict_proba',
                n_jobs=-1,
                passthrough=stack_config['use_features']
            )

        # Return single model if not stack ensemble
        return base_models.get(model_name)

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # threshold
        y_pred = (y_pred_proba > 0.3).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred_proba
        }
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        
        return metrics

class CrossValidationEvaluator:
    @staticmethod
    def perform_cross_validation(model, X, y, n_splits=5):
        try:
            # Initialize KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Calculate scores for different metrics
            accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
            precision_scores = cross_val_score(model, X, y, cv=kf, scoring='precision')
            recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall')
            f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
            
            # Compile results
            cv_results = {
                'accuracy': {
                    'mean': accuracy_scores.mean(),
                    'std': accuracy_scores.std(),
                    'min': accuracy_scores.min(),
                    'max': accuracy_scores.max(),
                    'scores': accuracy_scores
                },
                'precision': {
                    'mean': precision_scores.mean(),
                    'std': precision_scores.std(),
                    'min': precision_scores.min(),
                    'max': precision_scores.max(),
                    'scores': precision_scores
                },
                'recall': {
                    'mean': recall_scores.mean(),
                    'std': recall_scores.std(),
                    'min': recall_scores.min(),
                    'max': recall_scores.max(),
                    'scores': recall_scores
                },
                'f1': {
                    'mean': f1_scores.mean(),
                    'std': f1_scores.std(),
                    'min': f1_scores.min(),
                    'max': f1_scores.max(),
                    'scores': f1_scores
                }
            }
            
            return cv_results
            
        except Exception as e:
            print(f"Error in cross-validation: {str(e)}")
            return None

    @staticmethod
    def format_cv_results(results_dict):
        """Format cross-validation results into a DataFrame"""
        if results_dict is None:
            return pd.DataFrame()
            
        formatted_results = []
        for metric, values in results_dict.items():
            formatted_results.append({
                'Metric': metric.capitalize(),
                'Mean Score': f"{values['mean']:.3f}",
                'Std Dev': f"±{values['std']:.3f}",
                'Min Score': f"{values['min']:.3f}",
                'Max Score': f"{values['max']:.3f}"
            })
        
        return pd.DataFrame(formatted_results)

def train_all_models(X_train, X_test, y_train, y_test, selected_models, stack_config=None):
    results = {}
    
    for model_name in selected_models:
        try:
            # Get model
            model = ModelFactory.get_model(
                model_name, 
                random_state=42,
                stack_config=stack_config if model_name == 'STACK' else None
            )
            
            if model is None:
                print(f"Warning: Model {model_name} not found")
                continue
                
            # Train model
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            print(f"Evaluating {model_name}...")
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
            
            results[model_name] = {
                'model': model,
                'metrics': metrics
            }
            
            # Print initial results
            print(f"{model_name} Results:")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    return results