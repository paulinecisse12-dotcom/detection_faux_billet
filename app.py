import streamlit as st
import requests
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page avec thème marron/orange
st.set_page_config(
    page_title="Détecteur de Faux Billets", 
    page_icon="💵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec palette marron/orange
st.markdown("""
<style>
    .main {
        background-color: #FFF8F0;
    }
    .stApp {
        background-color: #FFF8F0;
    }
    .header {
        background-color: #8B4513;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFE4C4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #D2691E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-box {
        background-color: #FFEFD5;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px dashed #D2691E;
        text-align: center;
    }
    .success-box {
        background-color: #F5F5DC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #D2691E;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #FFEBCD;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #8B4513;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #D2691E;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8B4513;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #8B4513;
    }
    h1, h2, h3 {
        color: #8B4513;
    }
</style>
""", unsafe_allow_html=True)

# En-tête de l'application
st.markdown('<div class="header"><h1>💵 Détecteur de Faux Billets - Analyse par Lot</h1></div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>Cette application utilise un modèle de Machine Learning pour détecter les faux billets.</p>
    <p>Uploadez un fichier CSV contenant les mesures des billets à analyser.</p>
</div>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict-batch"

# Section pour uploader le fichier
st.markdown("## 📤 Upload du fichier CSV")
st.markdown("""
<div class="upload-box">
    <h3 style="color: #8B4513;">Déposez votre fichier ici</h3>
    <p>Le fichier doit contenir les colonnes: diagonal, height_left, height_right, margin_low, margin_up, length</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type="csv",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Aperçu du fichier
    df_preview = pd.read_csv(uploaded_file, sep=';')
    st.markdown("### Aperçu des données")
    st.dataframe(df_preview.head(), use_container_width=True)
    
    # Vérifier les colonnes
    required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    missing_columns = [col for col in required_columns if col not in df_preview.columns]
    
    if missing_columns:
        st.error(f"❌ Colonnes manquantes dans le fichier: {missing_columns}")
    else:
        if st.button("🚀 Lancer l'analyse des billets", type="primary", use_container_width=True):
            with st.spinner('Analyse en cours... Cela peut prendre quelques secondes'):
                try:
                    # Préparer le fichier pour l'envoi
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    
                    # Envoyer la requête à l'API
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Afficher les résultats principaux
                    st.markdown(f"""
                    <div class="success-box">
                        <h2 style="color: #8B4513; text-align: center;">✅ Analyse Terminée!</h2>
                        <p style="text-align: center; font-size: 1.2rem;">{result['statistics']['total_count']} billets analysés avec succès</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Créer un DataFrame avec les résultats
                    results_df = pd.DataFrame(result['results'])
                    
                    # ==================== VISUALISATIONS ====================
                    
                    # 1. KPI Cards avec style personnalisé
                    st.markdown("## 📊 Tableau de bord des résultats")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Total Billets</h3>
                            <p style="font-size: 2rem; color: #8B4513; font-weight: bold;">{result['statistics']['total_count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Billets Authentiques</h3>
                            <p style="font-size: 2rem; color: #2E8B57; font-weight: bold;">{result['statistics']['authentic_count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Faux Billets</h3>
                            <p style="font-size: 2rem; color: #DC143C; font-weight: bold;">{result['statistics']['fake_count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Taux de Faux Billets</h3>
                            <p style="font-size: 2rem; color: #D2691E; font-weight: bold;">{result['statistics']['fake_percentage']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 2. Répartition Authentique/Faux (Camembert)
                    st.markdown("### 📊 Répartition des billets")
                    fig_pie = px.pie(
                        names=['Authentiques', 'Faux'],
                        values=[result['statistics']['authentic_count'], result['statistics']['fake_count']],
                        color=['Authentiques', 'Faux'],
                        color_discrete_map={'Authentiques':'#2E8B57', 'Faux':'#DC143C'}
                    )
                    fig_pie.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#8B4513', width=2)),
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#8B4513")
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 3. Distribution de la confiance
                    #st.markdown("### 📈 Distribution du niveau de confiance")
                    
                    # Créer des labels pour les prédictions
                    #results_df['prediction_label'] = results_df['prediction'].apply(lambda x: 'Authentique' if x else 'Faux')
                    
                    #fig_hist = px.histogram(
                        #results_df, 
                        #x='confidence',
                        #color='prediction_label',
                        #nbins=20,
                        #color_discrete_map={'Authentique': '#2E8B57', 'Faux': '#DC143C'},
                        #labels={'confidence': 'Niveau de Confiance', 'count': 'Nombre de Billets'},
                        #opacity=0.8
                    #)
                    #fig_hist.update_layout(
                        #bargap=0.1,
                        #paper_bgcolor='rgba(0,0,0,0)',
                        #plot_bgcolor='rgba(0,0,0,0)',
                        #font=dict(color="#8B4513"),
                        #legend=dict(
                            #orientation="h",
                            #yanchor="bottom",
                            #y=1.02,
                            #xanchor="right",
                            #x=1
                        #)
                    #)
                    #st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 4. Tableau détaillé des résultats
                    st.markdown("### 📋 Résultats détaillés")
                    
                    # Formater l'affichage
                    display_df = results_df.copy()
                    display_df['prediction'] = display_df['prediction'].apply(lambda x: '✅ Authentique' if x else '❌ Faux')
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                    display_df['prob_false'] = display_df['prob_false'].apply(lambda x: f"{x:.1%}")
                    display_df['prob_true'] = display_df['prob_true'].apply(lambda x: f"{x:.1%}")
                    
                    # Appliquer le style aux lignes du tableau
                    def color_row(row):
                        if row['prediction'] == '✅ Authentique':
                            return ['background-color: #A5D6A7;'] * len(row)  # Vert très clair
                        else:
                            return ['background-color: #EF9A9A;'] * len(row)  # Rouge très clair
                    
                    styled_df = display_df.head(10).style.apply(color_row, axis=1)
                    
                    # Afficher seulement les 10 premiers pour l'aperçu
                    st.dataframe(styled_df, use_container_width=True)
                    
                    if len(display_df) > 10:
                        st.info(f"Affichage des 10 premiers résultats sur {len(display_df)}. Téléchargez le CSV complet pour voir tous les résultats.")
                    
                    # 5. Télécharger les résultats
                    st.markdown("### 💾 Téléchargement des résultats")
                    csv = results_df.to_csv(index=False, sep=';')
                    st.download_button(
                        label="📥 Télécharger tous les résultats (CSV)",
                        data=csv,
                        file_name="resultats_analyse_billets_complet.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Résumé exécutif
                    with st.expander("📋 Résumé Exécutif", expanded=True):
                        st.markdown(f"""
                        <div style="background-color: #FFEBCD; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #8B4513;">
                            <h3 style="color: #8B4513; margin-top: 0;">Résumé de l'analyse</h3>
                            <ul style="color: #8B4513;">
                                <li>🔍 <strong>Total analysé:</strong> {result['statistics']['total_count']} billets</li>
                                <li>✅ <strong>Authentiques:</strong> {result['statistics']['authentic_count']} billets ({result['statistics']['authentic_count']/result['statistics']['total_count']*100:.1f}%)</li>
                                <li>❌ <strong>Faux:</strong> {result['statistics']['fake_count']} billets ({result['statistics']['fake_percentage']:.1f}%)</li>
                                <li>🎯 <strong>Confiance moyenne:</strong> {result['statistics']['avg_confidence']:.1%}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Erreur de connexion à l'API : {e}")
                    st.info("Vérifiez que votre serveur FastAPI est lancé sur `http://127.0.0.1:8000`")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")
                    # Afficher le détail de l'erreur si disponible
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        error_detail = e.response.json()
                        st.error(f"Détail de l'erreur: {error_detail}")


# Section d'information et exemple
st.markdown("---")
st.markdown("## ℹ️ Instructions")

col_info, col_example = st.columns(2)

with col_info:
    st.markdown("""
    <div style="background-color: #FFF8DC; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #D2691E;">
        <h3 style="color: #8B4513;">Guide d'utilisation</h3>
        <ol style="color: #8B4513;">
            <li><strong>Format requis du fichier CSV:</strong>
                <ul>
                    <li><code>diagonal</code> (float) - Longueur diagonale</li>
                    <li><code>height_left</code> (float) - Hauteur gauche</li>
                    <li><code>height_right</code> (float) - Hauteur droite</li>
                    <li><code>margin_low</code> (float) - Marge basse</li>
                    <li><code>margin_up</code> (float) - Marge haute</li>
                    <li><code>length</code> (float) - Longueur</li>
                </ul>
            </li>
            <li><strong>Uploader le fichier</strong> dans la zone prévue ci-dessus</li>
            <li><strong>Cliquez sur 'Lancer l'analyse des billets'</strong> pour obtenir les résultats</li>
            <li><strong>Téléchargez les résultats</strong> si nécessaire</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col_example:
    st.markdown("""
    <div style="background-color: #FFEBCD; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #8B4513;">
        <h3 style="color: #8B4513;">Exemple de format</h3>
        <p style="color: #8B4513;">Téléchargez ce fichier exemple pour comprendre le format attendu:</p>
    </div>
    """, unsafe_allow_html=True)
    
    example_data = {
        'diagonal': [172.25, 171.80, 172.10],
        'height_left': [103.95, 104.20, 103.75],
        'height_right': [104.05, 103.90, 104.15],
        'margin_low': [4.15, 4.30, 4.05],
        'margin_up': [3.05, 2.95, 3.15],
        'length': [113.20, 112.95, 113.35]
    }
    example_df = pd.DataFrame(example_data)
    
    st.download_button(
        label="📥 Télécharger l'exemple (CSV)",
        data=example_df.to_csv(index=False, sep=';'),
        file_name="exemple_format_billets.csv",
        mime="text/csv"
    )

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B4513;">
    <p>Développé par Pauline Noëlie NDEBHANE CISSE</p>
    <p>Solution de détection de faux billets - Tous droits réservés © 2025</p>
</div>
""", unsafe_allow_html=True)