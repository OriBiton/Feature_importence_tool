import matplotlib
matplotlib.use('Agg')
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_image_from_figure(fig):
    """ ממיר את האובייקט matplotlib figure למחרוזת base64 כדי שניתן יהיה להציג אותו ישירות ב-HTML """
    buf = io.BytesIO()  # יצירת buffer בזיכרון
    fig.savefig(buf, format="png", bbox_inches="tight")  # שמירה ל-buffer בפורמט PNG
    buf.seek(0)  # חזרה להתחלה של ה-buffer
    encoded_string = base64.b64encode(buf.getvalue()).decode("utf-8")  # קידוד base64
    buf.close()  # סגירת ה-buffer
    return encoded_string