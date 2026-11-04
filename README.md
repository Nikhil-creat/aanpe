# AANPE v4.0 - Mars Rover Perception & Terrain Safety Engine
By Nikhil Chary Sriramoju. `index.html` = showcase site; `engine/` = Python project; `Dockerfile` at root.

## Publish on GitHub Pages
1. Create a repo (e.g. `aanpe`) on github.com, public, no README.
2. Unzip, then in the folder (Termux or PC):
   git init && git add . && git commit -m "AANPE v4.0"
   git branch -M main && git remote add origin https://github.com/Nikhil-creat/aanpe.git && git push -u origin main
3. Repo > Settings > Pages > Source: Deploy from a branch > main / (root) > Save.
4. Open https://nikhil-creat.github.io/aanpe/ after about a minute.

## Run the engine
cd engine && pip install -r requirements.txt && python model.py && python evaluate.py && python app.py
Docker: docker build -t aanpe . && docker run --gpus all -p 7860:7860 aanpe

## Chatbot
Open the site, tap "Rover AI", paste a free Groq or Gemini key (console.groq.com / aistudio.google.com).

Compose: docker compose up (dashboard :7860, API :8000, POST /predict)
