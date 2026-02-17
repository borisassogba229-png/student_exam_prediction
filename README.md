# Prédiction des Scores d'Examen des Étudiants

Projet de Machine Learning pour prédire les scores d'examen d'étudiants en fonction de leurs habitudes d'étude, santé mentale, et autres facteurs comportementaux.

## Objectif

Développer un modèle de régression capable de prédire avec précision le score d'examen (exam_score) d'un étudiant en analysant 25 variables explicatives liées à ses habitudes de vie et d'étude.

## Dataset

- *Source* : [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/amar5693/student-performance-dataset)
- *Auteur* : Amar Mandal
- *Nom du fichier* : ultimate_student_productivity_dataset_5000.csv
- *Taille* : 5000 étudiants
- *Variables* : 21 colonnes (20 features + 1 target)

### Variables principales :
- *Habitudes d'étude* : study_hours, self_study_hours, online_classes_hours
- *Santé & bien-être* : sleep_hours, mental_health_score, burnout_level
- *Activités* : social_media_hours, gaming_hours, screen_time_hours, exercise_minutes
- *Facteurs contextuels* : academic_level, part_time_job, upcoming_deadline, internet_quality
- *Scores* : productivity_score, focus_index
- *Cible* : exam_score

### Comment obtenir le dataset :
1. Téléchargez depuis [Kaggle](https://www.kaggle.com/datasets/amar5693/student-performance-dataset)
2. Placez le fichier CSV dans le dossier data/

## Méthodologie

### 1. Analyse Exploratoire (EDA)
- Statistiques descriptives complètes
- Analyse de la distribution de la variable cible
- Matrice de corrélation (heatmap)
- Détection et analyse des valeurs aberrantes (outliers)
- Visualisations (histogrammes, boxplots)

*Insights clés de l'EDA :*
- Distribution asymétrique de exam_score avec un pic important pour les faibles scores
- Corrélation très forte entre productivity_score (0.89) et exam_score
- Corrélation négative significative entre burnout_level (-0.41) et performance

### 2. Préparation des Données
- Suppression de student_id (identifiant non-informatif)
- One-Hot Encoding des variables catégorielles :
  - gender (Male, Female, Other)
  - academic_level (High School, Undergraduate, Postgraduate)
  - internet_quality (Good, Average, Poor)
- Normalisation avec StandardScaler (mean=0, std=1)
- Train/Test Split stratifié (80/20) avec random_state=42

### 3. Modélisation
Test comparatif de 3 approches de régression :

*a) Régression Linéaire Multiple*
- Baseline simple avec toutes les features (25)
- Relations linéaires directes

*b) Régression Polynomiale (degré 2)*
- Création de features polynomiales (350 features)
- Capture des relations non-linéaires
- Interactions entre variables

*c) Régression Lasso*
- Régularisation L1 pour sélection de features
- Optimisation de l'hyperparamètre alpha (0.001 à 5.0)
- Réduction du sur-apprentissage

## Résultats

### Comparaison des Modèles

| Modèle | R² Score | MAE | RMSE | Features Utilisées |
|--------|----------|-----|------|--------------------|
| Régression Linéaire | 0.8164 | 3.96 | 5.00 | 25 |
| Régression Polynomiale | 0.8131 | 4.04 | 5.05 | 350 |
| *Lasso (α=0.1)* | *0.8226* | *3.91* | *4.92* | *46* |

### Modèle Gagnant : Régression Lasso (alpha=0.1)

*Performance :*
- *R² = 0.8226* - Le modèle explique 82.26% de la variance des scores d'examen
- *MAE = 3.91* - Erreur moyenne absolue de ±3.91 points
- *RMSE = 4.92* - Faible erreur quadratique, peu de grosses erreurs

*Avantages :*
- Meilleure généralisation sur les données de test
- Sélection automatique des 46 features les plus pertinentes (sur 350)
- Évite le sur-apprentissage de la régression polynomiale
- Modèle parcimonieux et interprétable

## Variables les Plus Importantes

D'après l'analyse de corrélation avec exam_score :

### Impact Positif Fort
1. *productivity_score* (r=0.89) - Productivité globale
2. *focus_index* (r=0.75) - Capacité de concentration
3. *mental_health_score* (r=0.55) - Santé mentale
4. *study_hours* (r=0.51) - Heures d'étude
5. *sleep_hours* (r=0.23) - Qualité du sommeil

### Impact Négatif
1. *burnout_level* (r=-0.41) - Niveau d'épuisement
2. *upcoming_deadline* (r=-0.22) - Stress des échéances
3. *part_time_job* (r=-0.15) - Emploi à temps partiel
4. *screen_time_hours* (r=-0.13) - Temps d'écran total
5. *social_media_hours* (r=-0.11) - Temps sur réseaux sociaux

### Variables Peu Impactantes
- self_study_hours (r=0.08)
- exercise_minutes (r=0.04)
- online_classes_hours (r=0.00)
- caffeine_intake_mg (r=-0.08)

## Insights & Recommandations

### Pour améliorer les performances académiques :

*À Favoriser :*
- Développer des habitudes de productivité structurées
- Améliorer la concentration et minimiser les distractions
- Prendre soin de sa santé mentale (support, activités relaxantes)
- Maintenir des heures d'étude régulières et efficaces
- Dormir suffisamment (7-9h par nuit)

*À Éviter :*
- Le burnout - équilibrer travail et repos
- Le stress excessif lié aux deadlines multiples
- Temps excessif sur les écrans et réseaux sociaux
- Surcharge avec un emploi à temps partiel pendant les études intensives

### Observations surprenantes :
- L'auto-étude seule (self_study_hours) a peu d'impact - La qualité prime sur la quantité
- L'exercice physique n'a pas d'effet direct mesurable - Peut-être un effet indirect via la santé mentale
- La caféine a un léger effet négatif - Possiblement lié au stress ou au manque de sommeil

## Technologies Utilisées

- *Python 3.8+*
- *pandas* - Manipulation de données
- *numpy* - Calculs numériques
- *scikit-learn* - Modélisation ML
- *matplotlib* - Visualisations
- *seaborn* - Visualisations statistiques
- *Google Colab* - Environnement de développement

## Structure du Projet


student-exam-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── Student_Exam_Score_Prediction.ipynb
│
└── data/
    └── .gitkeep


## Installation & Utilisation

### Prérequis
- Python 3.8 ou supérieur
- pip

### 1. Cloner le repository
bash
git clone https://github.com/votre-username/student-exam-prediction.git
cd student-exam-prediction


### 2. Installer les dépendances
bash
pip install -r requirements.txt


### 3. Télécharger le dataset
1. Allez sur [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/amar5693/student-performance-dataset)
2. Téléchargez le fichier CSV
3. Placez-le dans data/ultimate_student_productivity_dataset_5000.csv

### 4. Ouvrir le notebook

*Option A : Jupyter Notebook (local)*
bash
jupyter notebook notebook/Student_Exam_Score_Prediction.ipynb


*Option B : Google Colab (en ligne)*
1. Allez sur [Google Colab](https://colab.research.google.com/)
2. Fichier → Ouvrir le notebook → GitHub
3. Collez l'URL de ce repository
4. Uploadez le dataset dans Colab

## Pipeline Complet


Données brutes (5000 × 21)
    ↓
Analyse Exploratoire (EDA)
    ↓
Nettoyage (suppression student_id)
    ↓
Feature Engineering (One-Hot Encoding)
    ↓
Normalisation (StandardScaler)
    ↓
Train/Test Split (80/20)
    ↓
Modélisation (3 modèles)
    ↓
Optimisation (tuning alpha)
    ↓
Évaluation finale
    ↓
Modèle optimal : Lasso (α=0.1)


## Métriques d'Évaluation

- *R² Score* : Proportion de variance expliquée (0 à 1)
- *MAE* : Erreur moyenne absolue
- *RMSE* : Erreur quadratique moyenne (pénalise les grosses erreurs)

## Améliorations Futures

- Cross-Validation K-Fold pour évaluation plus robuste
- Feature Engineering avancé
- Modèles Ensemble (Random Forest, Gradient Boosting, XGBoost)
- Deep Learning pour patterns complexes
- SHAP values pour interprétabilité
- Déploiement (API REST + interface Streamlit)
- A/B Testing de différentes stratégies

## Auteur : Boris ASSOGBA

- Étudiant en Data Science
- E-mail : borisassogba229@gmail.com
- GitHub : borisassogba229-png

## Licence

Ce projet est sous licence MIT.

## Remerciements

- Kaggle et Amar Mandal pour le dataset
- La communauté scikit-learn
- Google Colab pour l'infrastructure

---

Projet réalisé dans le cadre de ma formation en Data Science - Février 2026
```
