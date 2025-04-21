import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
import gc
import io
import base64



def fig_to_base64(fig):
    """ ממיר אובייקט matplotlib לפורמט Base64 לצורך הצגה ב-Django """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    gc.collect()
    return img_base64

def explain_model_with_shap(model, X_train):
    results = {}
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    
    if not isinstance(shap_values, shap.Explanation):
        shap_values = shap.Explanation(shap_values.values, base_values=shap_values.base_values, data=X_train)
    
    # 📊 SHAP Summary Plot
    fig_summary = plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    results['summary_plot'] = fig_to_base64(fig_summary)
    
    # 📊 Feature Importance Bar Chart
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)[:10]
    
    fig_importance = plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette='coolwarm')
    plt.xlabel("SHAP Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    results['feature_importance_plot'] = fig_to_base64(fig_importance)
    
    # 🔹 מציאת הפיצ'ר הכי חשוב
    most_important_feature = X_train.columns[np.argmax(feature_importance)]
    results['selected_feature'] = most_important_feature
    
    # 2️⃣ SHAP Dependence Bar Plot
    df_dependence = X_train[[most_important_feature]].copy()
    df_dependence['SHAP Value'] = shap_values.values[:, np.argmax(feature_importance)]
    
    fig_dependence = plt.figure(figsize=(8, 5))
    sns.barplot(x=most_important_feature, y='SHAP Value', data=df_dependence, errorbar=None)
    plt.title(f"SHAP Dependence Bar Plot - {most_important_feature}")
    results['dependence_plot'] = fig_to_base64(fig_dependence)
    
    # 🔹 חישוב אינטראקציות בין פיצ'רים
    shap_df = pd.DataFrame(shap_values.values, columns=list(X_train.columns))
    feature_importance_series = pd.Series(feature_importance)
    top_features = feature_importance_series.nlargest(100).index.tolist()
    top_features = [X_train.columns[i] for i in top_features]
    
    feature_pairs = list(combinations(top_features, 2))
    interaction_effects = []
    
    for f1, f2 in feature_pairs:
        combined_effect = np.abs(shap_df[f1] + shap_df[f2]).mean()
        separate_effect = np.abs(shap_df[f1]).mean() + np.abs(shap_df[f2]).mean()
        interaction_strength = combined_effect - separate_effect
        max_feature_effect = max(np.abs(shap_df[f1]).mean(), np.abs(shap_df[f2]).mean())
        relative_impact = (interaction_strength / max_feature_effect) * 100  
        interaction_effects.append((f1, f2, interaction_strength, relative_impact))
    
    interaction_df = pd.DataFrame(interaction_effects, columns=['Feature 1', 'Feature 2', 'SHAP Interaction Impact', 'Relative Impact (%)'])
    interaction_df = interaction_df.reindex(interaction_df['SHAP Interaction Impact'].sort_values(ascending=False).index)
    
    positive_interactions = interaction_df[interaction_df['Relative Impact (%)'] > 0.1]
    if positive_interactions.shape[0] == 0:
        results['interaction_plot'] = None
    else:
        fig_interactions = plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            x=interaction_df['SHAP Interaction Impact'][:15], 
            y=[f"{f1} & {f2}" for f1, f2 in zip(interaction_df['Feature 1'][:15], interaction_df['Feature 2'][:15])], 
            palette="coolwarm"
        )
        for i, (impact, percent) in enumerate(zip(interaction_df['SHAP Interaction Impact'][:15], interaction_df['Relative Impact (%)'][:15])):
            ax.text(impact, i, f"{percent:.1f}%", va='center', ha='left', fontsize=12, color='black')
        plt.xlabel("SHAP Interaction Strength")
        plt.ylabel("Feature Pairs")
        plt.title("Top 15 SHAP Feature Interactions (with Relative Impact)")
        results['interaction_plot'] = fig_to_base64(fig_interactions)

    # 🔹 חישוב החשיבות היחסית
    total_shap_value = feature_importance.sum()
    relative_importance = feature_importance / total_shap_value * 100
    
    # 🔹 קביעת קטגוריית ההשפעה לכל משתנה
    def categorize_impact(value):
        if value >= 10:
            return "Strong impact"
        elif value >= 1:
            return "Med impact"
        else:
            return "Low impact"

    impact_categories = np.vectorize(categorize_impact)(relative_importance)

    # יצירת טבלה עם החשיבות היחסית וקטגוריית ההשפעה
    feature_impact_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Relative Importance (%)': relative_importance,
        'Impact Category': impact_categories
    }).sort_values(by='Relative Importance (%)', ascending=False)
    feature_impact_df=feature_impact_df[feature_impact_df['Relative Importance (%)']>0.05]
    # הוספת הטבלה ל-results
    results['feature_impact_table'] = feature_impact_df.to_html()
    
    del shap_values, importance_df, explainer, X_train
    gc.collect()
    
    return results