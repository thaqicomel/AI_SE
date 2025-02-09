import streamlit as st
import pandas as pd
import numpy as np
from data_utils import load_data, preprocess_data, split_data, get_feature_importance
from model_utils import train_all_models, ModelFactory
from viz_utils import plot_roc_curves, plot_metrics_comparison, plot_cv_results
import plotly.graph_objects as go
import plotly.express as px
from model_utils import CrossValidationEvaluator

# Set page config
st.set_page_config(
    page_title="Cervical Cancer Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_stack_ensemble_ui():
    """Create UI for stack ensemble configuration"""
    st.sidebar.subheader("Stack Ensemble Configuration")
    
    # Get available models
    base_models_dict = ModelFactory.get_base_models()
    meta_learners_dict = ModelFactory.get_meta_learners()
    
    # Select base models
    selected_base_models = st.sidebar.multiselect(
        "Select Base Models",
        options=list(base_models_dict.keys()),
        default=['RF', 'GB', 'ET'],  # Changed default to available models
        help="Select models to use as base learners in the stack ensemble",
        format_func=lambda x: base_models_dict[x][0]
    )
    
    # Select meta-learner
    meta_learner = st.sidebar.selectbox(
        "Select Meta-Learner",
        options=list(meta_learners_dict.keys()),
        index=0,
        format_func=lambda x: meta_learners_dict[x][0],
        help="Select the model to use as the meta-learner"
    )
    
    # Cross-validation folds
    cv_folds = st.sidebar.slider(
        "Cross-validation Folds",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of folds for cross-validation during stacking"
    )
    
    # Use original features
    use_features = st.sidebar.checkbox(
        "Include Original Features",
        value=True,
        help="Include original features along with base model predictions"
    )
    
    # Show selected configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stack Ensemble Summary")
    st.sidebar.markdown(f"**Base Models:** {', '.join(selected_base_models)}")
    st.sidebar.markdown(f"**Meta-Learner:** {meta_learners_dict[meta_learner][0]}")
    st.sidebar.markdown(f"**CV Folds:** {cv_folds}")
    st.sidebar.markdown(f"**Using Original Features:** {'Yes' if use_features else 'No'}")
    
    return {
        'base_models': selected_base_models,
        'meta_learner': meta_learner,
        'cv_folds': cv_folds,
        'use_features': use_features
    }
def main():
    st.title("Cervical Cancer Prediction Model Comparison")
    
    # Sidebar
    st.sidebar.header("Imputation and Model Settings")
    
    # Imputation method selection with detailed parameters
    imputation_method = st.sidebar.selectbox(
        "Select imputation method",
        ['xgboost', 'lightgbm', 'soft', 'knn', 'pca', 'mice'],
        help="Choose the method for handling missing values"
    )
    
    # Imputation parameters based on selected method
    imputation_params = {}
    
    if imputation_method == 'knn':
        st.sidebar.subheader("KNN Imputer Parameters")
        imputation_params['n_neighbors'] = st.sidebar.slider(
            "Number of neighbors (k)",
            min_value=1,
            max_value=20,
            value=5
        )
        imputation_params['weights'] = st.sidebar.selectbox(
            "Weight function",
            ['uniform', 'distance']
        )
        
    elif imputation_method == 'pca':
        st.sidebar.subheader("PCA Imputer Parameters")
        imputation_params['max_iter'] = st.sidebar.slider(
            "Maximum iterations",
            min_value=10,
            max_value=200,
            value=100
        )
        imputation_params['tol'] = st.sidebar.number_input(
            "Convergence tolerance",
            min_value=1e-8,
            max_value=1e-4,
            value=1e-6,
            format="%.1e"
        )
        imputation_params['n_components'] = st.sidebar.slider(
            "Explained variance ratio",
            min_value=0.1,
            max_value=1.0,
            value=0.95,
            step=0.05
        )
        
    elif imputation_method in ['mice', 'xgboost', 'lightgbm']:
        st.sidebar.subheader(f"{imputation_method.upper()} Parameters")
        imputation_params['max_iter'] = st.sidebar.slider(
            "Maximum iterations",
            min_value=5,
            max_value=50,
            value=10
        )
        imputation_params['random_state'] = 42
        
    elif imputation_method == 'soft':
        st.sidebar.subheader("Soft Impute Parameters")
        imputation_params['max_iter'] = st.sidebar.slider(
            "Maximum iterations",
            min_value=10,
            max_value=200,
            value=100
        )
        imputation_params['threshold'] = st.sidebar.number_input(
            "Threshold",
            min_value=1e-8,
            max_value=1e-2,
            value=1e-5,
            format="%.1e"
        )
    
    # Model Selection Section
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    
    # Choose between single models or stack ensemble
    model_type = st.sidebar.radio(
        "Select Model Type",
        options=['Single Models', 'Stack Ensemble'],
        help="Choose between individual models or stack ensemble"
    )
    
    stack_config = None
    if model_type == 'Single Models':
        available_models = [
            'LR', 'DT', 'RF', 'XGB', 'SVC', 
            'ET', 'CAT', 'ADA', 'SGD', 'GB'
        ]
        
        model_descriptions = {
            'LR': 'Logistic Regression',
            'DT': 'Decision Tree',
            'RF': 'Random Forest',
            'XGB': 'XGBoost',
            'SVC': 'Support Vector Classifier',
            'ET': 'Extra Trees',
            'CAT': 'CatBoost',
            'ADA': 'AdaBoost',
            'SGD': 'Stochastic Gradient Descent',
            'GB': 'Gradient Boosting'
        }
        
        selected_models = st.sidebar.multiselect(
            "Select models to compare",
            available_models,
            default=['RF', 'XGB', 'CAT'],
            format_func=lambda x: f"{x} - {model_descriptions[x]}"
        )
    else:  # Stack Ensemble
        stack_config = create_stack_ensemble_ui()
        if not stack_config['base_models']:
            st.warning("Please select at least one base model for the stack ensemble.")
            return
        selected_models = ['STACK']
    
    # Data split ratio
    st.sidebar.markdown("---")
    st.sidebar.header("Data Split Settings")
    train_percentage = st.sidebar.slider(
        "Training Data Percentage",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.1
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    try:
        # Load and preprocess data
        df = load_data('cervical-cancer_csv.csv')
        
        # Data overview
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Dataset Shape:", df.shape)
        with col2:
            st.write("Number of Features:", df.shape[1] - 1)
        with col3:
            st.write("Target Distribution:")
            target_dist = pd.DataFrame({
                'Class': ['Negative', 'Positive'],
                'Count': df['Biopsy'].value_counts().values,
                'Percentage': (df['Biopsy'].value_counts(normalize=True) * 100).round(2).values
            })
            st.dataframe(target_dist)
        
        # Missing values analysis
        st.header("Missing Values Analysis")
        missing_stats = df.isnull().sum()
        if missing_stats.any():
            missing_df = pd.DataFrame({
                'Feature': missing_stats.index,
                'Missing Count': missing_stats.values,
                'Missing Percentage': (missing_stats.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
                'Missing Count', ascending=False
            )
            
            # Display missing values statistics
            st.dataframe(missing_df)
            
            # Plot missing values distribution
            fig = px.bar(
                missing_df,
                x='Feature',
                y='Missing Percentage',
                title='Missing Values Distribution',
                labels={'Missing Percentage': '% of Missing Values'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        
        # Preprocess data with selected imputation method
        with st.spinner('Preprocessing data...'):
            X_scaled, y, scaler, imputer = preprocess_data(
                df,
                imputation_method=imputation_method,
                **imputation_params
            )
        
        # Split data
        with st.spinner('Splitting data...'):
            X_train, X_test, y_train, y_test = split_data(
                X_scaled, 
                y, 
                train_percentage=train_percentage
            )
        
        # Model training progress
        st.header("Model Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train and evaluate models
        results = {}
        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((idx + 1) / len(selected_models))
            
            model_results = train_all_models(
                X_train, X_test, y_train, y_test, [model_name],
                stack_config=stack_config if model_name == 'STACK' else None
            )
            results.update(model_results)
        
        progress_bar.empty()
        status_text.text("Training completed!")
        
        # Display results
        st.header("Model Evaluation Results")
        
        # ROC Curves
        st.subheader("ROC Curves Comparison")
        st.plotly_chart(plot_roc_curves(results), use_container_width=True)
        
        # Model Metrics Comparison
        st.subheader("Model Performance Metrics")
        fig_metrics, metrics_df = plot_metrics_comparison(results)
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.dataframe(metrics_df)
        
        # Cross-validation Analysis
        st.header("Cross-validation Analysis")
        if st.checkbox("Show Cross-validation Results"):
            cv_results = {}
            for model_name, result in results.items():
                model_cv = CrossValidationEvaluator.perform_cross_validation(
                    result['model'],
                    X_scaled,
                    y
                )
                if model_cv:
                    cv_results[model_name] = model_cv
                    
                    st.subheader(f"{model_name} Cross-validation Results")
                    cv_df = CrossValidationEvaluator.format_cv_results(model_cv)
                    st.dataframe(cv_df)
                    
                    fig = plot_cv_results(model_cv, model_name)
                    st.plotly_chart(fig)
        
        # Feature Importance Analysis
        st.header("Feature Importance Analysis")
        if st.checkbox("Show Feature Importance"):
            for model_name, result in results.items():
                if hasattr(result['model'], 'feature_importances_') or hasattr(result['model'], 'coef_'):
                    importance_df = get_feature_importance(result['model'], X_scaled.columns)
                    if importance_df is not None:
                        st.subheader(f"{model_name} Feature Importance")
                        fig = px.bar(
                            importance_df.head(10),
                            x='importance',
                            y='feature',
                            title=f'Top 10 Important Features - {model_name}',
                            orientation='h'
                        )
                        st.plotly_chart(fig)
        
        # Download predictions
        st.header("Download Predictions")
        for model_name, result in results.items():
            predictions_df = pd.DataFrame({
                'True_Values': y_test,
                'Predicted_Probabilities': result['metrics']['predictions'],
                'Predicted_Labels': result['metrics']['predictions'] > 0.5
            })
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label=f'Download {model_name} Predictions',
                data=csv,
                file_name=f'{model_name.lower()}_predictions.csv',
                mime='text/csv'
            )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and try again.")

if __name__ == "__main__":
    main()