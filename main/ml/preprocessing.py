import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
from scipy import stats



def count_and_ID(df):
    #count
    count_col=[]
    for col in df.columns:
        if col.lower()=='count':
            count_col.append(col)
    if len(count_col)>0:
        count_col=count_col[0]
        df = df.loc[df.index.repeat(df[count_col])].reset_index(drop=True)
        df=df.drop(count_col,axis=1)

    #id   
    id_col=[]
    for col in df.columns:
        if pd.Series([col]).str.contains(r'(^id$|(?:_| )id$|Id)', case=False, regex=True).any():
            id_col.append(col)
    if len(id_col)>0:
            df[id_col] = df[id_col].astype('str')
    
    return df

def handle_outliers(df, z_threshold=3, iqr_multiplier=2, alpha=0.05):
    """מזהה ומטפל בערכים חריגים על בסיס Z-Score (אם הנתונים נורמליים) או IQR (אם לא)"""
    
    numeric_cols = df.select_dtypes(include=['number']).columns  # רק עמודות מספריות

    for column in numeric_cols:
        stat, p_value = stats.shapiro(df[column].dropna())  # בודק האם ההתפלגות נורמלית
        
        # 📌 אם הנתונים מתפלגים נורמלית → נשתמש ב-Z-Score
        if p_value > alpha:
            print(f"✅ עמודה {column} מתפלגת נורמלית → נשתמש ב-Z-Score")
            
            z_scores = stats.zscore(np.nan_to_num(df[column]))  # חישוב Z-Score בלי להסיר ערכים חסרים
            df.loc[abs(z_scores) > z_threshold, column] = np.nan  # הפיכת חריגים ל-NaN
        
        # 📌 אם הנתונים לא מתפלגים נורמלית → נשתמש ב-IQR
        else:
            print(f"⚠️ עמודה {column} לא מתפלגת נורמלית → נשתמש ב-IQR")
    
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
    
            df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan  # מחליף ב-NaN

    return df

def missing_values(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    object_cols = df.select_dtypes(include=['object']).columns

    for col in df.columns.copy():  # משתמשים ב-copy כדי למנוע בעיות מחיקה תוך כדי לולאה
        # 📌 שלב 1: הסרת עמודות עם יותר מ-50% ערכים חסרים
        if df[col].isnull().mean() >= 0.5:
            df.drop(columns=[col], inplace=True)
            print(f'Column {col} has been removed')
            continue

        # 📌 שלב 2: טיפול בעמודות מספריות עם פחות מ-50% ערכים חסרים
        elif col in numeric_cols:
            mean = df[col].mean()
            if mean == 0:
                mean = 1e-9  # למניעת חלוקה באפס
            std_ratio = df[col].std() / mean  # יחס סטיית תקן לממוצע

            if std_ratio > 1:
                df[col] = df[col].fillna(df[col].median())  # ✅ שינוי ערכים ישירות
                print(f'Column {col} handled by median')
            else:
                df[col] = df[col].fillna(df[col].mean())  # ✅ שינוי ערכים ישירות
                print(f'Column {col} handled by mean')

        # 📌 שלב 3: טיפול בעמודות קטגוריות עם 30%-50% ערכים חסרים
        elif col in object_cols and 0.3 < df[col].isnull().mean() < 0.5:
            mode_value = df[col].mode()[0]
            mode_count = df[col].value_counts().get(mode_value, 0)
            mode_ratio = mode_count / len(df)

            if mode_ratio >= 0.3:
                df[col] = df[col].fillna(mode_value)  # ✅ שינוי ערכים ישירות
                print(f'Column {col} handled by mode')
            else:
                df.drop(columns=[col], inplace=True)
                print(f'Column {col} has been removed')

        # 📌 שלב 4: טיפול בשאר העמודות הקטגוריות
        elif col in object_cols:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)  # ✅ שינוי ערכים ישירות
            print(f'Column {col} handled by mode')

    return df

def prepare_data(df):
    print('count_and_ID:')
    df=count_and_ID(df)
    print('handle_outliers:')
    df=handle_outliers(df, z_threshold=3, iqr_multiplier=2, alpha=0.05)
    print('missing_values:')
    df=missing_values(df)
    return df