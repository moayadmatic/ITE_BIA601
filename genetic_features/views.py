from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, View
from django.contrib import messages
from .models import Dataset, FeatureSelection
import pandas as pd
import numpy as np
from .genetic_algorithm import GeneticFeatureSelector
from .advanced_genetic_algorithm import AdvancedGeneticFeatureSelector

class HomeView(ListView):
    model = Dataset
    template_name = 'genetic_features/home.html'
    context_object_name = 'datasets'

class UploadDatasetView(View):
    def get(self, request):
        return render(request, 'genetic_features/upload.html')
    
    def post(self, request):
        if 'file' not in request.FILES:
            messages.error(request, 'Please select a file to upload')
            return redirect('upload')
        
        file = request.FILES['file']
        name = request.POST.get('name', file.name)
        
        try:
            # Validate file
            df = pd.read_csv(file)
            if len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns")
            
            dataset = Dataset.objects.create(name=name, file=file)
            messages.success(request, 'Dataset uploaded successfully')
            return redirect('dataset_detail', pk=dataset.pk)
            
        except Exception as e:
            messages.error(request, f'Error uploading file: {str(e)}')
            return redirect('upload')

class DatasetDetailView(DetailView):
    model = Dataset
    template_name = 'genetic_features/dataset_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.get_object()
        
        # Read dataset to get basic info
        df = pd.read_csv(dataset.file.path)
        context['n_features'] = len(df.columns) - 1  # Exclude target column
        context['n_samples'] = len(df)
        context['preview'] = df.head().to_html(classes='table table-striped')
        context['columns'] = df.columns.tolist()  # Add columns list for target selection
        
        # Get feature selection results for this dataset
        context['selections'] = FeatureSelection.objects.filter(dataset=dataset).order_by('-created_at')
        
        # Debug: print selections count
        print(f"DEBUG: Dataset {dataset.id} has {context['selections'].count()} selections")
        
        return context

class RunFeatureSelectionView(View):
    def post(self, request, pk):
        dataset = Dataset.objects.get(pk=pk)
        if request.method == 'POST':
            # Get parameters from form
            target_column = request.POST.get('target_column', 'itemDescription')
            chromosome_type = request.POST.get('chromosome_type', 'adaptive')
            population_size = int(request.POST.get('population_size', 50))
            generations = int(request.POST.get('generations', 20))
            
            try:
                # Load dataset
                df = pd.read_csv(dataset.file.path)
                
                # Validate target column exists
                if target_column not in df.columns:
                    messages.error(request, f'Target column "{target_column}" not found in dataset.')
                    return redirect('dataset_detail', pk=pk)
                
                # Check if we have too many unique values in target column
                if df[target_column].nunique() > 100:
                    messages.warning(request, f'Too many unique values in {target_column}. Taking top 100 most frequent values.')
                    # Keep only top 100 most frequent items
                    top_items = df[target_column].value_counts().nlargest(100).index
                    df = df[df[target_column].isin(top_items)]
                
                # Preprocess the data - handle date columns dynamically
                date_columns = []
                for col in df.columns:
                    if col.lower() in ['date', 'time', 'datetime'] or 'date' in col.lower():
                        try:
                            # Try to convert to datetime
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            if not df[col].isna().all():  # If conversion was successful
                                date_columns.append(col)
                                # Extract date features
                                date_features = pd.concat([
                                    df[col].dt.year.rename(f'{col}_Year'),
                                    df[col].dt.month.rename(f'{col}_Month'),
                                    df[col].dt.day.rename(f'{col}_Day'),
                                    df[col].dt.dayofweek.rename(f'{col}_DayOfWeek')
                                ], axis=1)
                                # Add date features to dataframe
                                df = df.join(date_features)
                        except:
                            pass  # Skip if conversion fails
                
                # Drop original date columns
                if date_columns:
                    df = df.drop(date_columns, axis=1)
                    messages.info(request, f'Converted {len(date_columns)} date columns to numerical features.')
                
                # Encode categorical variables
                from sklearn.preprocessing import LabelEncoder
                
                # Create encoders dictionary to store the mapping
                encoders = {}
                
                # Encode categorical columns
                for column in df.select_dtypes(include=['object']).columns:
                    encoders[column] = LabelEncoder()
                    df[column] = encoders[column].fit_transform(df[column])
                
                # Prepare features and target using selected target column
                X = df.drop([target_column], axis=1).values
                y = df[target_column].values
                
                # Add message about dataset size and target column
                messages.info(request, f'Processing {len(df)} samples with {X.shape[1]} features.')
                messages.info(request, f'Target column: {target_column} ({len(np.unique(y))} unique classes)')
                
                # Get feature names (excluding target column)
                feature_cols = [col for col in df.columns if col != target_column]
                
                # Choose algorithm type based on data size
                if X.shape[1] > 10 or len(y) > 10000:
                    # Use advanced algorithm for large datasets
                    selector = AdvancedGeneticFeatureSelector(
                        X, y,
                        population_size=population_size,
                        generations=generations,
                        max_features_ratio=0.3,
                        cv_folds=3,
                        scoring_method='balanced',
                        early_stopping_patience=5,
                        sample_size=5000 if len(y) > 5000 else None,
                        chromosome_type=chromosome_type
                    )
                    selected_features, fitness_score, logbook, detailed_results = selector.run()
                    
                    # Get selected feature names
                    selected_feature_names = [feature_cols[i] for i, selected in enumerate(selected_features) if selected]
                    
                    # Save detailed results
                    selection = FeatureSelection.objects.create(
                        dataset=dataset,
                        selected_features=selected_feature_names,
                        fitness_score=float(fitness_score),
                        generation=detailed_results['generations_run'],
                        detailed_results=detailed_results,
                        algorithm_type='advanced'
                    )
                    
                    messages.success(request, f'Selected {detailed_results["n_selected_features"]} features out of {X.shape[1]} features successfully')
                    messages.info(request, f'Improvement ratio: {detailed_results["improvement_over_random"]:.3f}')
                    
                    # Debug: print saved selection
                    print(f"DEBUG: Saved selection ID {selection.id} for dataset {dataset.id}")
                    print(f"DEBUG: Selected features: {selected_feature_names}")
                    print(f"DEBUG: Algorithm type: {selection.algorithm_type}")
                    
                else:
                    # Use simple algorithm for small datasets
                    selector = GeneticFeatureSelector(X, y, chromosome_type=chromosome_type)
                    selected_features, fitness_score, logbook = selector.run()
                    
                    # Get selected feature names
                    selected_feature_names = [feature_cols[i] for i, selected in enumerate(selected_features) if selected]
                    
                    # Save results with original feature names
                    selection = FeatureSelection.objects.create(
                        dataset=dataset,
                        selected_features=selected_feature_names,
                        fitness_score=float(fitness_score),
                        generation=len(logbook),
                        algorithm_type='basic'
                    )
                    
                    messages.success(request, 'Feature selection completed successfully')
                    
                    # Debug: print saved selection
                    print(f"DEBUG: Saved basic selection ID {selection.id} for dataset {dataset.id}")
                    print(f"DEBUG: Selected features: {selected_feature_names}")
                    print(f"DEBUG: Algorithm type: {selection.algorithm_type}")
                
            except Exception as e:
                messages.error(request, f'Error during feature selection: {str(e)}')
        
        # Redirect to results page instead of dataset detail
        if 'selection' in locals():
            return redirect('selection_results', dataset_id=pk, selection_id=selection.id)
        else:
            return redirect('dataset_detail', pk=pk)

class SelectionResultsView(DetailView):
    model = FeatureSelection
    template_name = 'genetic_features/results.html'
    context_object_name = 'selection'
    pk_url_kwarg = 'selection_id'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['dataset'] = Dataset.objects.get(pk=self.kwargs['dataset_id'])
        return context
