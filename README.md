# Prédiction des prix immobiliers à Paris
Projet avancé Big Data & IA pour prédire les prix des appartements à Paris avec les données DVF 2024 (~26,000 transactions).

## Fonctionnalités
- **Nettoyage** : Agrégation, gestion des valeurs manquantes, features (prix/m², distance au centre, clusters).
- **Visualisations** : Tableau de bord Dash avec fond animé et design parisien, carte interactive, prix au m².
- **Modélisation** : LightGBM avec GridSearchCV et SHAP (RMSE ~ [insert RMSE], R² ~ [insert R²]).

## Outils
Python, Pandas, Scikit-learn, LightGBM, SHAP, Plotly, Dash, Tailwind CSS.

## Résultats
- Visualisations : [Tableau de bord animé](dashboard.py), [Carte clusters](map_clusters.html), [Prix au m²](boxplot_price_per_m2.png).
- Notebook : [Paris_Price_Prediction.ipynb](Paris_Price_Prediction.ipynb).

## Live Demo
Access the interactive dashboard online: [https://paris-price-prediction.onrender.com/](https://paris-price-prediction.onrender.com/)