from flask import Flask, request, render_template, send_file, redirect
import os
import pandas as pd
import joblib
import re
import string

# === CONFIGURATION FLASK ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# === CHARGEMENT DES MODÈLES ===
vectorizer = joblib.load("models/vectorizer.joblib")
clf_l1 = joblib.load("models/xgb_level1.joblib")
le_level1 = joblib.load("models/label_encoder_level1.joblib")
le_level2 = joblib.load("models/label_encoder_level2.joblib")
models_level2 = joblib.load("models/models_level2_dict.joblib")
le_level2_per_l1 = joblib.load("models/le_level2_per_l1_dict.joblib")

# === NETTOYAGE TEXTE ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# === PRÉDICTION HIÉRARCHIQUE ===
def predict_hierarchical_fast(desc):
    x_input = vectorizer.transform([desc])

    # Prédiction Level1
    l1_pred_enc = clf_l1.predict(x_input)[0]
    try:
        l1_label = le_level1.inverse_transform([l1_pred_enc])[0]
    except ValueError:
        return "Label1 inconnu", "Label2 inconnu"

    # Prédiction Level2
    if l1_label not in models_level2:
        return l1_label, "Pas de sous-catégorie disponible"

    clf_l2 = models_level2[l1_label]
    le_local = le_level2_per_l1[l1_label]

    l2_pred_local = clf_l2.predict(x_input)[0]
    try:
        l2_label_text = le_local.inverse_transform([l2_pred_local])[0]
    except ValueError:
        l2_label_text = "Label2 inconnu"

    return l1_label, l2_label_text


# === PAGE PRINCIPALE ===
@app.route("/")
def index():
    return render_template("index.html")


# === CLASSIFICATION PAR TEXTE ===
@app.route("/classify_text", methods=["POST"])
def classify_text():
    desc = request.form.get("input_text")  # correspond à name="input_text" dans index.html
    if not desc:
        return redirect("/")

    desc_cleaned = clean_text(desc)
    l1_label, l2_label = predict_hierarchical_fast(desc_cleaned)

    return render_template("index.html", prediction={"level1": l1_label, "level2": l2_label})


# === CLASSIFICATION PAR FICHIER ===
@app.route("/classify_file", methods=["POST"])
def classify_file():
    file = request.files["file"]
    if not file:
        return redirect("/")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Lecture du fichier
    if filename.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        return "Format non supporté. Utilise .xlsx ou .csv"

    if "description" not in df.columns:
        return "Le fichier doit contenir une colonne 'description'"

    # Prédictions
    cleaned = df["description"].astype(str).apply(clean_text)
    level1_preds, level2_preds = [], []

    for desc in cleaned:
        l1, l2 = predict_hierarchical_fast(desc)
        level1_preds.append(l1)
        level2_preds.append(l2)

    df["Level 1 (catégorie)"] = level1_preds
    df["Level 2 (sous-catégorie)"] = level2_preds

    output_path = os.path.join(RESULT_FOLDER, f"result_{filename.split('.')[0]}.xlsx")
    df.to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True)


# === LANCEMENT DE L'APPLICATION ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # nécessaire pour Render
    app.run(host="0.0.0.0", port=port, debug=True)
