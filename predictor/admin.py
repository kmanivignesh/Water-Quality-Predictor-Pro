from django.contrib import admin
from .models import WaterQualityPrediction

@admin.register(WaterQualityPrediction)
class WaterQualityPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'prediction', 'created_at')
    list_filter = ('prediction', 'created_at')
    search_fields = ('prediction',)
    ordering = ('-created_at',)