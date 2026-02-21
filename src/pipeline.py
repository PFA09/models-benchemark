import json
import os
import pandas as pd
from tqdm import tqdm
from jiwer import cer
from src.models_wrapper import get_model

def load_config(config_path="config/config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    config = load_config()
    os.makedirs(config["output_folder"], exist_ok=True)
    
    # Charger les métadonnées
    with open(config["dataset_path"], 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_results = []
    summary_metrics = {}

    for model_cfg in config["models"]:
        print(f"\n--- Initialisation du modèle : {model_cfg['id']} ---")
        model = get_model(model_cfg)
        
        predictions = []
        ground_truths = []
        correct_count = 0

        print(f"Inférence sur {len(dataset)} fichiers...")
        for item in tqdm(dataset):
            audio_path = os.path.join(config["audio_folder"], item["file"])
            true_label = str(item["label"]).upper()
            
            # Gestion de l'erreur dans les données (audio062 a un label vide "W" dans "type")
            if not true_label: continue 
            
            try:
                pred_label = model.predict(audio_path)
            except Exception as e:
                print(f"Erreur sur {item['file']}: {e}")
                pred_label = ""

            predictions.append(pred_label)
            ground_truths.append(true_label)
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct_count += 1

            all_results.append({
                "model_id": model_cfg["id"],
                "file": item["file"],
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "type": item.get("type", "Unknown")
            })

        # Calcul des métriques globales pour ce modèle
        # CER calcule la distance d'édition entre caractères
        error_rate = cer(ground_truths, predictions)
        accuracy = correct_count / len(dataset)

        summary_metrics[model_cfg["id"]] = {
            "Accuracy": round(accuracy * 100, 2),
            "CER": round(error_rate * 100, 2)
        }
        print(f"Résultats {model_cfg['id']} -> Accuracy: {accuracy*100:.2f}% | CER: {error_rate*100:.2f}%")

    # 1. Sauvegarde des prédictions détaillées (pour analyse des erreurs)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(config["output_folder"], "detailed_predictions.csv"), index=False)

    # 2. Sauvegarde du résumé exécutif
    with open(os.path.join(config["output_folder"], "evaluation_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"\nÉvaluation terminée. Résultats sauvegardés dans '{config['output_folder']}/'")

if __name__ == "__main__":
    run_evaluation()