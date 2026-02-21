# PFA09 - Benchmark ASR (Reconnaissance Vocale pour la Dysarthrie)

Ce dépôt contient le pipeline d'évaluation automatisé pour comparer les performances de différents modèles de reconnaissance automatique de la parole (ASR) "Zéro-Shot" sur des données vocales dysarthriques (lettres isolées et chiffres).

Ce module correspond à la validation technique de notre chaîne de traitement (Phase V0/V1 du projet PFA).

## 📋 Prérequis

* **Python** : `3.10` strictement recommandé.
* **OS** : Compatible Windows / Linux / macOS.
* **Matériel** : Un GPU (CUDA) est recommandé pour accélérer l'inférence, mais le code fonctionne également sur CPU.

## ⚙️ Installation

1. **Créer un environnement virtuel** (recommandé) :
```bash
conda create --name PFA python=3.10

```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt

```



## 📁 Architecture du Projet

Le projet respecte une architecture modulaire séparant la configuration, les données (soumises au RGPD) et le code source :

```text
benchmark_asr/
├── config/
│   └── config.json          # Paramétrage des modèles et des chemins
├── data/                    # ⚠️ NON VERSIONNÉ (Données sensibles)
│   ├── audios/              # Fichiers .wav du locuteur
│   └── dataset.json         # Étiquetage et métadonnées (Lettres/Chiffres)
├── results/                 # Rapports générés automatiquement (CER, Accuracy)
├── src/
│   ├── models_wrapper.py    # Logique d'inférence spécifique (Wav2Vec2, Whisper...)
│   └── pipeline.py          # Orchestrateur d'évaluation
├── requirements.txt
└── README.md

```

## 🚀 Utilisation

1. **Préparer les données** :
Placez vos enregistrements `.wav` dans `data/audios/` et votre fichier de description dans `data/dataset.json`.
2. **Configurer les modèles** :
Éditez le fichier `config/config.json` pour ajouter ou modifier les modèles de la plateforme Hugging Face que vous souhaitez évaluer.
3. **Lancer l'évaluation** :
Exécutez le pipeline depuis la racine du projet :
```bash
python -m src.pipeline

```



## 📊 Résultats générés

Après exécution, le dossier `results/` contiendra :

* `evaluation_summary.json` : Résumé des métriques globales (Accuracy, CER) par modèle.
* `detailed_predictions.csv` : Fichier détaillé permettant d'analyser les erreurs spécifiques de prédiction lettre par lettre.

## 🔒 Note sur la confidentialité (RGPD)

Les données vocales collectées auprès du client sont des données sensibles. Le dossier `data/` et son contenu doivent obligatoirement être ignorés par Git (voir `.gitignore`) et ne jamais être poussés sur un dépôt distant public.