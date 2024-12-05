import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Titre de l'application
st.title('Prédiction de la Satisfaction des Employés')

# Instructions
st.sidebar.info(
    """
    ### Instructions :
    1. Téléchargez un fichier Excel contenant vos données.
    2. Sélectionnez une variable cible (ex. : Satisfaction).
    3. Visualisez les distributions et les corrélations, puis testez les prédictions.
    """
)

# Chargement des données
st.sidebar.header('Charger les données')
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    # Charger le fichier Excel avec pandas
    data = pd.read_excel(uploaded_file)
    st.write("Données chargées", data.head())

    # Gestion des valeurs manquantes
    # data.fillna(0, inplace=True)

    # Afficher les statistiques descriptives
    st.write("Résumé des données (Statistiques)", data.describe())

    # Visualisation des distributions
    st.subheader("Distribution des variables")
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    selected_column = st.selectbox("Sélectionner une variable", numerical_columns)

    # Tracer l'histogramme de la colonne sélectionnée
    bins = st.slider("Choisissez le nombre de bins pour l'histogramme", min_value=5, max_value=50, value=20)
    fig, ax = plt.subplots()
    data[selected_column].hist(bins=bins, ax=ax)
    ax.set_title(f"Distribution de {selected_column}")
    st.pyplot(fig)

    # Corrélation entre les variables
    st.subheader("Matrice de corrélation")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Sélection des variables et modélisation pour les prédictions
    st.sidebar.header('Sélection des variables')
    target_variable = st.sidebar.selectbox("Sélectionner la variable cible (Satisfaction)", data.columns)

    # Sélectionner les variables explicatives
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    # Préparer les données
    X_numerical = X.select_dtypes(include=[np.number])  # Isoler les colonnes numériques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Séparer les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Entraînement du modèle Random Forest pour la classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Afficher la matrice de confusion (pour classification, ici affichage de la régression)
    st.subheader("Graphique des prédictions vs Réel")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Valeurs prédites')
    ax.set_title('Prédictions vs Réel')
    st.pyplot(fig)

    # Afficher les résultats de la classification
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Précision du modèle : {accuracy:.2f}")
    st.write("Rapport de classification :")
    st.text(classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Vérités réelles")
    st.pyplot(fig)

    # Prédictions interactives
    st.subheader("Prédictions interactives")
    new_data = []

    for col in X.columns:
        if col in ['DERNIER_AVACEMENT_EN_GRADE', 'DATENAISSA', 'DATEENGAGE']:
            continue
        if col == 'ETATCIVIL':
            options = ['Célibataire', 'Marié(e)', 'Divorcé(e)', 'Veuf/Veuve']
            value = st.selectbox(f"Entrez une valeur pour {col}", options)
            new_data.append(options.index(value))  # Convertir en index numérique

        elif col == 'SEXE':  # Sexe : 0 = Féminin, 1 = Masculin
            value = st.radio(f"Entrez une valeur pour {col}", options=['Féminin', 'Masculin'])
            new_data.append(0 if value == 'Féminin' else 1)

        elif col == 'ENFANT':
            value = st.number_input(f"Entrez une valeur pour {col}", value=0, step=1, min_value=0, max_value=20)
            new_data.append(value)

        elif col == 'GRADE':
            options = sorted(X[col].unique())
            value = st.selectbox(f"Entrez une valeur pour {col}", options)
            new_data.append(value)

        elif col == 'CITE':
            options = sorted(X[col].unique())
            value = st.selectbox(f"Entrez une valeur pour {col}", options)
            new_data.append(value)

        elif col in X_numerical.columns:  # Autres colonnes numériques
            value = st.number_input(f"Entrez une valeur pour {col}", value=0.0)
            new_data.append(value)

        else:  # Par défaut, champ texte
            value = st.text_input(f"Entrez une valeur pour {col}")
            new_data.append(value)

    # Effectuer une prédiction pour de nouvelles données
    if st.button("Faire une prédiction"):
        if len(new_data) == len(X_numerical.columns):
            # Appliquer le scaler
            new_data_scaled = scaler.transform([new_data])

            # Appliquer PCA
            new_data_pca = pca.transform(new_data_scaled)

            # Faire la prédiction avec le modèle
            prediction = model.predict(new_data_pca)

            # Si la variable cible est binaire, adapter la sortie
            if target_variable in ["Satisfait", "Non satisfait"]:  # Exemple de variables cibles binaires
                satisfaction = "Satisfait" if prediction[0] == 1 else "Non satisfait"
            else:
                # Si la variable cible est une échelle de satisfaction (par exemple, de 1 à 5)
                # Contraindre la prédiction à la plage 1 à 5
                predicted_score = max(1, min(5, int(prediction[0])))  # S'assurer que la prédiction est entre 1 et 5
                score_mapping = {
                    5: 1,
                    4: 2,
                    3: 3,
                    2: 4,
                    1: 5
                }

                # Convertir le score prédit en fonction du dictionnaire
                if predicted_score != 0 or predicted_score != 1:
                    converted_score = score_mapping.get(predicted_score,
                                                        "Score invalide")  # Valeur par défaut pour gestion des erreurs
                    satisfaction = f"Score de satisfaction : {converted_score} (sur une échelle de 1 à 5)"
                else:
                    satisfaction = "Satisfait" if prediction[0] == 1 else "Non satisfait"

            # Afficher la prédiction
            st.write(f"Prédiction de satisfaction : {satisfaction}")

        else:
            st.error(
                f"Erreur : Les données saisies ne correspondent pas au format attendu ({len(X_numerical.columns)} colonnes).")
