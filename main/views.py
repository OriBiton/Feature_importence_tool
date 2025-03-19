from django.shortcuts import render
from .forms import UploadFileForm,ChooseTargetColumn,IsVisualization
import pandas as pd
from .ml.preprocessing import prepare_data
import json
from django.shortcuts import redirect
from .ml.visualization import get_image_from_figure
import pickle
import base64
from .ml.explainability import explain_model_with_shap
from .ml.training import build_and_train_model, detect_problem_type, encoding_and_normalizing
import os
from django.conf import settings
def home(request):
    return render(request, 'main/home.html', {})


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            try:
                df = pd.read_csv(csv_file)
                cleaned_data=prepare_data(df)
                request.session['data_types'] = {col: str(dtype) for col, dtype in cleaned_data.dtypes.items()}
                request.session['data'] = cleaned_data.to_json()
                table = cleaned_data.head().to_html()
                return render(request, 'main/results.html', {'table': table})
            except Exception as e:
                print('Error:', e)
                return render(request, 'main/upload.html', {'form': form, 'error': str(e)})

    else:
        form = UploadFileForm()
    return render(request, 'main/upload.html', {'form': form})

def results(response):
    return render(response, 'main/results.html', {})

def choosing_target_and_is_visual(request):
    data_json = request.session.get('data')
    if data_json:
        df = pd.read_json(data_json)
        data_types = request.session.get('data_types')

        # שחזור סוגי הנתונים
        for col, dtype_str in data_types.items():
            df[col] = df[col].astype(dtype_str)
        request.session['data'] = df.to_json()
        columns = df.columns.tolist()

    if request.method == 'POST':
        form_target = ChooseTargetColumn(request.POST)
        form_show = IsVisualization(request.POST)

        # עדכון בחירות דינמיות לשדות
        form_target.fields['target_column'].choices = [(col, col) for col in columns]

        if form_target.is_valid() and form_show.is_valid():
            # שמירת הבחירות ל-session או לשימוש בצעדים הבאים
            request.session['selected_target_column'] = form_target.cleaned_data['target_column']
            request.session['show_visual'] = form_show.cleaned_data['show_visual']

            # מעבר לפי בחירת המשתמש
            if form_show.cleaned_data['show_visual']:
                return redirect('show_visuals')  # שם ה־URL להגדרות שלך
            else:
                return redirect('train_models')  # שם ה־URL האחר

    else:
        form_target = ChooseTargetColumn()
        form_target.fields['target_column'].choices = [(col, col) for col in columns]
        form_show = IsVisualization()

    return render(request, 'main/target_col.html', {
        'form_target': form_target,
        'form_show': form_show,
    })

from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from .forms import ColumnSelectionForm, ColumnPairForm
import io
def show_visuals(request):
    data_json = request.session.get('data')
    if data_json:
        df = pd.read_json(data_json)
        data_types = request.session.get('data_types')

        # שחזור סוגי הנתונים
        for col, dtype_str in data_types.items():
            df[col] = df[col].astype(dtype_str)
        request.session['data'] = df.to_json()
        columns = df.columns.tolist()

        column_form = ColumnSelectionForm(columns=columns)
        pair_form = ColumnPairForm(columns=columns)
        visuals = []

        # 1️⃣ טיפול בבחירת עמודה אחת
        if request.method == 'POST' and 'single_column' in request.POST:
            column_form = ColumnSelectionForm(request.POST, columns=columns)
            if column_form.is_valid():
                selected_col = column_form.cleaned_data['column']
                fig, ax = plt.subplots(figsize=(30, 20))  # גודל גרף גדול

                if df[selected_col].dtype == 'object':
                    sns.countplot(x=df[selected_col], hue=df[selected_col], palette="pastel", legend=False, ax=ax)
                    ax.set_title(f'Distribution of {selected_col}', fontsize=60)

                else:
                    sns.histplot(df[selected_col], kde=True, bins=30, color="blue", ax=ax)
                    ax.set_title(f'Distribution of {selected_col}', fontsize=60)

                # הגדלת תוויות הצירים
                ax.set_xlabel(selected_col, fontsize=55)
                ax.set_ylabel("Count" if df[selected_col].dtype == 'object' else "Frequency", fontsize=55)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                # הגדלת הכיתובים על הצירים
                ax.tick_params(axis='x', labelsize=50)
                ax.tick_params(axis='y', labelsize=50)

                    

                visuals.append({'title': f'Distribution of {selected_col}', 'img': get_image_from_figure(fig)})

        # 2️⃣ טיפול בבחירת שתי עמודות
        if request.method == 'POST' and 'column_pair' in request.POST:
            pair_form = ColumnPairForm(request.POST, columns=columns)
            if pair_form.is_valid():
                col1 = pair_form.cleaned_data['column1']
                col2 = pair_form.cleaned_data['column2']
                fig, ax = plt.subplots(figsize=(30, 20))  # גודל גרף גדול

                if df[col1].dtype == 'object' and df[col2].dtype == 'object':
                    sns.heatmap(pd.crosstab(df[col1], df[col2]), annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title(f'Heatmap of how many times {col1} and {col2} appear together', fontsize=30)

                elif df[col1].dtype == 'object' and df[col2].dtype != 'object' or df[col1].dtype != 'object' and df[col2].dtype == 'object':
                    sns.boxplot(x=df[col1], y=df[col2], palette="viridis", ax=ax)
                    ax.set_title(f'Boxplot of {col1} by {col2}', fontsize=60)

                elif df[col1].dtype != 'object' and df[col2].dtype != 'object':
                    sns.regplot(x=df[col1], y=df[col2], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
                    ax.set_title(f'Scatter + Regression of {col1} by {col2}', fontsize=60)

                # הגדלת תוויות הצירים
                ax.set_xlabel(col1, fontsize=50)
                ax.set_ylabel(col2, fontsize=50)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                # הגדלת הכיתובים על הצירים
                ax.tick_params(axis='x', labelsize=50)
                ax.tick_params(axis='y', labelsize=50)


                visuals.append({'title': f'the connection between {col1} and {col2} ','img': get_image_from_figure(fig)})

        return render(request, 'main/visualization.html', {
            'column_form': column_form,
            'pair_form': pair_form,
            'visuals': visuals,
        })


def train_models(request):
    data_json = request.session.get('data')
    if data_json:
        df = pd.read_json(data_json)
        data_types = request.session.get('data_types')

        # שחזור סוגי הנתונים
        for col, dtype_str in data_types.items():
            df[col] = df[col].astype(dtype_str)
        request.session['data'] = df.to_json()
        columns = df.columns.tolist()

    # קבלת העמודה המטרה
    target_col = request.session.get('selected_target_column')

    
    # זיהוי סוג בעיה
    problem_type = detect_problem_type(df[target_col])

    # עיבוד נתונים
    df, columns = encoding_and_normalizing(df, target_col)

    # בניית ואימון המודל
    best_model, trained_features, results,X_train = build_and_train_model(df, problem_type, target_col)

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    model_path = os.path.join(settings.MEDIA_ROOT, 'trained_model.pkl')
    # שמירת המודל כקובץ
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # שמירת הנתיב והרשימה ב-session לשימוש בהמשך
    request.session['trained_model_path'] = model_path
    request.session['trained_features'] = trained_features
    request.session['X_train'] = X_train.to_json()
    print(f"✅ Model saved at: {model_path}")
    return render(request, 'main/train_results.html', {
        'results': results,
        'target_column': target_col,
    })
    
def shap_analysis(request):
    data_json = request.session.get('data')
    X_train_json = request.session.get('X_train')
    model_path = request.session.get('trained_model_path')
    trained_features = request.session.get('trained_features')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    df = pd.read_json(data_json)
    X_train=pd.read_json(X_train_json)
    results = explain_model_with_shap(model, X_train)
    context = {
        "summary_plot": results.get("summary_plot"),
        "feature_importance_plot": results.get("feature_importance_plot"),
        "dependence_plot": results.get("dependence_plot"),
        "interaction_plot": results.get("interaction_plot"),
        "selected_feature": results.get("selected_feature"),
        "feature_impact_table": results.get("feature_impact_table"),
    }
    
    return render(request, "main/shap_results.html", context)
        



