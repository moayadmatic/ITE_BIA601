from django.db import models
import json

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class FeatureSelection(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    selected_features = models.JSONField()
    fitness_score = models.FloatField()
    generation = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    detailed_results = models.JSONField(null=True, blank=True)  # للنتائج المفصلة من الخوارزمية المتقدمة
    algorithm_type = models.CharField(max_length=50, default='basic')  # نوع الخوارزمية المستخدمة
    
    class Meta:
        ordering = ['-fitness_score']

    def __str__(self):
        return f"{self.dataset.name} - Generation {self.generation}"
    
    def get_improvement_percentage(self):
        """حساب نسبة التحسن"""
        if self.detailed_results and 'improvement_over_random' in self.detailed_results:
            return self.detailed_results['improvement_over_random'] * 100
        return None
    
    def get_selection_ratio(self):
        """حساب نسبة الميزات المختارة"""
        if self.detailed_results and 'selection_ratio' in self.detailed_results:
            return self.detailed_results['selection_ratio'] * 100
        return None
