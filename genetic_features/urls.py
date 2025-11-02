from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomeView.as_view(), name='dataset_list'),
    path('upload/', views.UploadDatasetView.as_view(), name='upload'),
    path('dataset/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('dataset/<int:pk>/run/', views.RunFeatureSelectionView.as_view(), name='run_feature_selection'),
    path('dataset/<int:dataset_id>/results/<int:selection_id>/', views.SelectionResultsView.as_view(), name='selection_results'),
]