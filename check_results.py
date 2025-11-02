#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'feature_selection.settings')
django.setup()

from genetic_features.models import FeatureSelection, Dataset

print("=== فحص النتائج في قاعدة البيانات ===")
print(f"عدد النتائج الكلي: {FeatureSelection.objects.count()}")
print(f"عدد مجموعات البيانات: {Dataset.objects.count()}")

print("\n=== آخر 5 نتائج ===")
for fs in FeatureSelection.objects.all().order_by('-id')[:5]:
    print(f"ID: {fs.id}")
    print(f"Dataset: {fs.dataset.name}")
    print(f"Selected Features: {fs.selected_features}")
    print(f"Fitness Score: {fs.fitness_score}")
    print(f"Algorithm Type: {getattr(fs, 'algorithm_type', 'غير محدد')}")
    print(f"Created: {fs.created_at}")
    print("---")

print("\n=== فحص Dataset ID 2 ===")
try:
    dataset = Dataset.objects.get(id=2)
    print(f"Dataset Name: {dataset.name}")
    selections = FeatureSelection.objects.filter(dataset=dataset)
    print(f"عدد النتائج لهذا Dataset: {selections.count()}")
    
    for sel in selections:
        print(f"- ID: {sel.id}, Features: {sel.selected_features}, Score: {sel.fitness_score}")
except Dataset.DoesNotExist:
    print("Dataset ID 2 غير موجود")
