from . import views
from django.urls import path
urlpatterns = [
    path('', views.home,name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('results/',views.results, name='results'),
    path('target_col/',views.choosing_target_and_is_visual, name='target_col'),
    path('visualizations/',views.show_visuals, name='show_visuals'),
    path('train_models/', views.train_models, name='train_models'),
    path('shap/', views.shap_analysis, name='shap'),
    

]