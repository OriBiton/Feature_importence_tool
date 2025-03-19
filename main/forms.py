from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')

class ChooseTargetColumn(forms.Form):
    target_column = forms.ChoiceField(label='Select Target Column')

class IsVisualization(forms.Form):
    show_visual = forms.BooleanField(label='Show Visualizations', required=False)


from django import forms

class ColumnSelectionForm(forms.Form):
    column = forms.ChoiceField(label="Select a column", choices=[])

    def __init__(self, *args, columns=None, **kwargs):
        super().__init__(*args, **kwargs)  # קריאה ל-Form המקורי
        if columns:
            self.fields['column'].choices = [(col, col) for col in columns]  # יצירת אפשרויות מתוך רשימת העמודות

class ColumnPairForm(forms.Form):
    column1 = forms.ChoiceField(label="Select the first column", choices=[])
    column2 = forms.ChoiceField(label="Select the second column", choices=[])

    def __init__(self, *args, columns=None, **kwargs):
        super().__init__(*args, **kwargs)  # קריאה ל-Form המקורי
        if columns:
            choices = [(col, col) for col in columns]  # יצירת אפשרויות
            self.fields['column1'].choices = choices
            self.fields['column2'].choices = choices
