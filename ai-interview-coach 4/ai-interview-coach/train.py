"""Simulate learners, compare 5 classifiers, save the best one.
Task: predict whether a candidate answers the NEXT question correctly."""
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = ["topic_acc", "recent3", "n_topic", "overall_acc", "difficulty"]
rng = np.random.default_rng(42)

def simulate(n_learners=400, n_topics=5, n_attempts=60):
    rows = []
    for L in range(n_learners):
        skill = rng.normal(0, 1, n_topics); learn = rng.uniform(0.01, 0.08)
        hist = [[] for _ in range(n_topics)]; allh = []
        for _ in range(n_attempts):
            t = rng.integers(n_topics); d = rng.integers(1, 3)
            h = hist[t]
            rows.append(dict(learner=L, topic_acc=(sum(h)+1)/(len(h)+2),
                recent3=(sum(h[-3:])+1)/(len(h[-3:])+2), n_topic=len(h),
                overall_acc=(sum(allh)+1)/(len(allh)+2), difficulty=d, y=0))
            p = 1/(1+np.exp(-1.6*(skill[t] + learn*len(h) - 0.7*(d-1))))
            ok = int(rng.random() < p); rows[-1]["y"] = ok
            h.append(ok); allh.append(ok)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = simulate(); df.to_csv("simulated_attempts.csv", index=False)
    # split by LEARNER so no person appears in both train and test
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=1).split(df, groups=df.learner))
    Xtr, ytr, Xte, yte = df.iloc[tr][FEATURES], df.iloc[tr].y, df.iloc[te][FEATURES], df.iloc[te].y
    models = {
      "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=1),
      "Naive Bayes": GaussianNB(),
      "SVM": SVC(probability=True, random_state=1),
      "Logistic Regression": LogisticRegression(max_iter=1000),
      "Random Forest": RandomForestClassifier(200, max_depth=8, random_state=1)}
    res, fitted = [], {}
    for name, m in models.items():
        pipe = make_pipeline(StandardScaler(), m)
        pipe.fit(Xtr.iloc[:6000] if name == "SVM" else Xtr, ytr.iloc[:6000] if name == "SVM" else ytr)
        pr = pipe.predict(Xte); pp = pipe.predict_proba(Xte)[:, 1]; fitted[name] = pipe
        res.append(dict(model=name, accuracy=accuracy_score(yte, pr), precision=precision_score(yte, pr),
                        recall=recall_score(yte, pr), f1=f1_score(yte, pr), roc_auc=roc_auc_score(yte, pp)))
    out = pd.DataFrame(res).round(3).sort_values("roc_auc", ascending=False)
    print(out.to_string(index=False)); out.to_csv("model_comparison.csv", index=False)
    best = out.iloc[0].model; print("\nBest model (by ROC-AUC):", best)
    joblib.dump(dict(model=fitted[best], features=FEATURES, name=best), "model.joblib")
