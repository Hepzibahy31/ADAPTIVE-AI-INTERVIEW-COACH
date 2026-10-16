# Adaptive AI Technical Interview Coach (mascot: Quill the hedgehog)

Run:
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000

Optional: set ANTHROPIC_API_KEY for Claude-written hedgehog feedback.
    Windows:  set ANTHROPIC_API_KEY=your_key
    Mac/Linux: export ANTHROPIC_API_KEY=your_key

Files:
    app.py               Flask backend (adaptive question picking, feedback, progress API)
    templates/index.html Front end (hedgehog mascot, voice, mastery bars)

Train ML first (optional, model.joblib is included): python train.py
Results: model_comparison.csv
The hedgehog asks 5 ML interview questions, then shows score out of 5 and what to improve.
