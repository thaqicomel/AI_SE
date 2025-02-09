import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_feature_importance(feature_importance_df, model_name):
    """Plot feature importance"""
    if feature_importance_df is None or feature_importance_df.empty:
        return None
        
    fig = px.bar(
        feature_importance_df.head(10),
        x='importance',
        y='feature',
        title=f'Top 10 Feature Importance - {model_name}',
        orientation='h'
    )
    fig.update_layout(height=400)
    return fig

def plot_confusion_matrix(conf_matrix, model_name):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400
    )
    return fig

def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    for model_name, result in results.items():
        metrics = result['metrics']
        fig.add_trace(go.Scatter(
            x=metrics['fpr'],
            y=metrics['tpr'],
            name=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})',
            mode='lines'
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    return fig

def plot_metrics_comparison(results):
    """Plot comparison of metrics across models with vibrant colors"""
    metrics_dict = {}
    
    for model_name, result in results.items():
        metrics = result['metrics']
        metrics_dict[model_name] = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1']
        }
    
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Define vibrant colors for each metric
    colors = {
        'Accuracy': '#007FFF',    # Vivid blue
        'Precision': '#FF0000',   # Bright red
        'Recall': '#00FF00',      # Bright green
        'F1 Score': '#FFD700'     # Golden yellow
    }
    
    # Create bar plot with vibrant colors
    fig = go.Figure()
    for metric in metrics_df.columns:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df.index,
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto',
            marker_color=colors[metric]  # Apply the vibrant colors
        ))
    
    fig.update_layout(
        title='Model Metrics Comparison',
        barmode='group',
        height=500,
        # Update the overall appearance
        plot_bgcolor='white',      # White background
        paper_bgcolor='white',     # White surrounding
        font=dict(size=12),        # Slightly larger font
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            orientation="h"
        )
    )
    
    return fig, metrics_df

def plot_cv_results(cv_results, model_name):
    """Plot cross-validation results with error bars"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig = go.Figure()
    
    # Define colors for different metrics
    colors = {
        'accuracy': '#007FFF',    # Vivid blue
        'precision': '#FF0000',   # Bright red
        'recall': '#00FF00',      # Bright green
        'f1': '#FFD700'          # Golden yellow
    }
    
    for metric in metrics:
        if metric in cv_results:
            result = cv_results[metric]
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=[model_name],
                y=[result['mean']],
                error_y=dict(
                    type='data',
                    array=[result['std']],
                    visible=True
                ),
                marker_color=colors[metric]
            ))
    
    fig.update_layout(
        title=f'{model_name} Cross-Validation Scores',
        yaxis_title='Score',
        barmode='group',
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        yaxis=dict(range=[0, 1]),  # Set y-axis range from 0 to 1
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            orientation="h"
        )
    )
    
    return fig