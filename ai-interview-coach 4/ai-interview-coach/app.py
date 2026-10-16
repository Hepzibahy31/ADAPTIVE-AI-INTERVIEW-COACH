import os, random
import pandas as pd, joblib
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
ROUND = 5  # questions per interview round

# topic, question, options, correct index, explanation
def Q(i, t, q, o, a, e): return {"id": i, "topic": t, "q": q, "o": o, "a": a, "e": e}
BANK = [
 Q(1,"Learning Types","Which type of learning uses labeled data?",["Unsupervised","Supervised","Reinforcement","Clustering"],1,"Supervised learning trains on inputs with known labels."),
 Q(2,"Learning Types","K-Means is an example of?",["Supervised learning","Unsupervised learning","Regression","Reinforcement learning"],1,"K-Means groups unlabeled data into clusters."),
 Q(3,"Learning Types","Predicting a house price is which task?",["Classification","Clustering","Regression","Ranking"],2,"Predicting a continuous number is regression."),
 Q(4,"Overfitting","A model has 99% train accuracy but 60% test accuracy. What is this?",["Underfitting","Overfitting","Perfect fit","Data leakage only"],1,"It memorised the training data and fails on new data."),
 Q(5,"Overfitting","Which helps reduce overfitting?",["More complex model","Regularization","Less training data","Removing the test set"],1,"Regularization penalises complexity. More data and cross-validation also help."),
 Q(6,"Overfitting","What does cross-validation do?",["Speeds up training","Estimates performance on unseen data","Adds features","Removes outliers"],1,"It trains and tests on several splits to estimate generalisation."),
 Q(7,"Evaluation Metrics","Which metric answers: of all predicted positives, how many were correct?",["Recall","Precision","Accuracy","ROC-AUC"],1,"Precision = true positives / predicted positives."),
 Q(8,"Evaluation Metrics","For rare-disease detection, which metric matters most?",["Accuracy","Recall","Speed","R-squared"],1,"Recall: you must not miss real positive cases. Accuracy is misleading on imbalanced data."),
 Q(9,"Evaluation Metrics","F1-score is the?",["Mean of precision and recall","Harmonic mean of precision and recall","Product of accuracy and recall","Max of precision and recall"],1,"F1 is the harmonic mean, so it is low if either one is low."),
 Q(10,"Algorithms","Random Forest is an ensemble of?",["Linear models","Decision trees","Neural networks","Clusters"],1,"It averages many decision trees trained on random samples."),
 Q(11,"Algorithms","Logistic Regression is used for?",["Regression","Classification","Clustering","Compression"],1,"Despite the name, it outputs a probability for classification."),
 Q(12,"Algorithms","Naive Bayes assumes features are?",["Correlated","Independent given the class","Normalised","Sorted"],1,"The 'naive' assumption is conditional independence of features."),
 Q(13,"Data Preparation","Why scale features before SVM?",["It adds data","SVM is distance-based and sensitive to scale","It removes labels","It is not needed"],1,"Unscaled large-range features dominate distance calculations."),
 Q(14,"Data Preparation","Fitting the scaler on the full dataset before splitting causes?",["Faster training","Data leakage","Underfitting","Nothing"],1,"Test information leaks into training. Fit on train only."),
 Q(15,"Data Preparation","How do you handle class imbalance?",["Ignore it","Resampling or class weights","Delete the minority class","Use only accuracy"],1,"Use oversampling, undersampling, or class weights, and judge with F1 or recall."),
]
TOPICS = sorted({b["topic"] for b in BANK})
TIPS = {"Learning Types":"Revise supervised vs unsupervised vs reinforcement learning, with an example of each.",
 "Overfitting":"Revise overfitting vs underfitting, regularization, and cross-validation.",
 "Evaluation Metrics":"Revise precision, recall, F1, ROC-AUC, and when accuracy misleads.",
 "Algorithms":"Revise how Decision Tree, Naive Bayes, SVM, Logistic Regression, and Random Forest work.",
 "Data Preparation":"Revise feature scaling, data leakage, and handling imbalanced classes."}

try: ML = joblib.load("model.joblib")
except Exception: ML = None  # run train.py first; falls back to simple accuracy
STATE = {}

def sess():
    sid = request.headers.get("X-Session") or "default"
    return STATE.setdefault(sid, {"stats": {t: [0, 0] for t in TOPICS}, "recent": {t: [] for t in TOPICS}, "round": [], "asked": []})

def p_correct(s, t, d=2):
    r, n = s["stats"][t]
    if not ML: return (r + 1) / (n + 2)
    rec = s["recent"][t][-3:]; tot = sum(v[0] for v in s["stats"].values()); cnt = sum(v[1] for v in s["stats"].values())
    x = pd.DataFrame([dict(topic_acc=(r+1)/(n+2), recent3=(sum(rec)+1)/(len(rec)+2), n_topic=n,
                           overall_acc=(tot+1)/(cnt+2), difficulty=d)])[ML["features"]]
    return float(ML["model"].predict_proba(x)[0, 1])  # trained model: P(next answer correct)

def pick(s):
    used = [b["topic"] for b in BANK if b["id"] in s["asked"]]
    # weakest topic first; topics already asked this round get a penalty so the round covers more ground
    topic = min(TOPICS, key=lambda t: p_correct(s, t) + 0.3 * used.count(t) + random.random() * 0.05)
    pool = [b for b in BANK if b["topic"] == topic and b["id"] not in s["asked"]]
    return random.choice(pool or [b for b in BANK if b["id"] not in s["asked"]])

def hedgehog_feedback(item, ok, chosen):
    prompt = (f"You are Quill, a cheerful hedgehog interview coach. In max 2 short spoken sentences, give feedback. "
              f"Question: {item['q']} Correct: {item['o'][item['a']]} Student chose: {item['o'][chosen]}. Explain: {item['e']}")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=120, messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        except Exception: pass
    return f"Spiky-good! That's right. {item['e']}" if ok else f"Ooh, prickly one! The answer is {item['o'][item['a']]}. {item['e']}"

@app.get("/")
def index(): return render_template("index.html")

@app.post("/api/start")
def start():
    s = sess(); s["round"] = []; s["asked"] = []
    return jsonify({"total": ROUND})

@app.get("/api/next")
def nxt():
    s = sess()
    if len(s["round"]) >= ROUND: return jsonify({"done": True})
    b = pick(s); s["asked"].append(b["id"])
    return jsonify({"done": False, "number": len(s["round"]) + 1, "total": ROUND, **{k: b[k] for k in ("id", "topic", "q", "o")}})

@app.post("/api/answer")
def answer():
    s = sess(); d = request.json
    item = next(b for b in BANK if b["id"] == d["id"]); ok = d["choice"] == item["a"]
    s["stats"][item["topic"]][0] += ok; s["stats"][item["topic"]][1] += 1; s["recent"][item["topic"]].append(int(ok))
    s["round"].append({"topic": item["topic"], "q": item["q"], "ok": ok, "yours": item["o"][d["choice"]], "right": item["o"][item["a"]], "why": item["e"]})
    return jsonify({"correct": ok, "answer": item["a"], "feedback": hedgehog_feedback(item, ok, d["choice"]),
                    "done": len(s["round"]) >= ROUND, "results": [r["ok"] for r in s["round"]]})

@app.get("/api/summary")
def summary():
    s = sess(); r = s["round"]; score = sum(x["ok"] for x in r)
    weak = sorted({x["topic"] for x in r if not x["ok"]})
    if not weak: speech = f"Amazing! You got {score} out of {len(r)}. No weak areas this round. Try again for new questions!"
    else: speech = f"You got {score} out of {len(r)}. Work on {' and '.join(weak)}. Check the tips on screen."
    return jsonify({"score": score, "total": len(r), "questions": r, "improve": [{"topic": t, "tip": TIPS[t]} for t in weak], "speech": speech})

if __name__ == "__main__": app.run(debug=True)
