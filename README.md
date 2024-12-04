# Travail Pratique de Datamining - Prédiction de la Satisfaction des Employés

Ce projet de **Datamining** vise à prédire la satisfaction des employés à partir de leurs données personnelles et professionnelles en utilisant des techniques de machine learning. L'objectif principal est de prédire si un employé est **satisfait** ou **non satisfait** en fonction de diverses caractéristiques.

## Fonctionnalités

- **Chargement des données** : Téléchargement des données au format **Excel** et affichage des premières lignes du fichier.
- **Nettoyage des données** : Préparation des données brutes, gestion des valeurs manquantes et transformation des variables.
- **Analyse exploratoire des données (EDA)** : Visualisation des distributions, matrices de corrélation et des relations entre les variables.
- **Modélisation prédictive** : Utilisation d'un modèle de classification binaire **Random Forest** pour prédire la satisfaction des employés.
- **Réduction de dimension** : Application de **PCA** pour réduire la dimensionnalité et optimiser les performances du modèle.
- **Application interactive** : Interface interactive avec **Streamlit** permettant :
    - Le chargement des données.
    - L'affichage des résultats d'analyse.
    - La prédiction de la satisfaction sur de nouvelles données.

## Technologies Utilisées

- **Python** : Langage principal utilisé pour le projet.
- **Pandas** : Manipulation et nettoyage des données.
- **Scikit-learn** : Modélisation et prétraitement des données.
- **Matplotlib & Seaborn** : Visualisation des données.
- **Streamlit** : Création de l'application web interactive pour l'interface utilisateur.

## Prérequis

- Python 3.7+
- Les bibliothèques suivantes doivent être installées :
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - streamlit

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/BaldoM/tp-datamining.git
   cd tp-datamining
