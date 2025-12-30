```
# EL PV Cell Extractor (Streamlit)

This repository contains a Streamlit app that extracts PV cells from EL PV module images and builds an image+mask dataset suitable for training segmentation models.

Files:
- streamlit_app.py — the Streamlit app.
- requirements.txt — Python dependencies.
- apt.txt — system packages required on Streamlit Community Cloud (fixes libGL import issues).

Deploy to Streamlit Community Cloud (recommended):
1. Commit & push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app" → choose the repository, branch (usually "main"), and set the "Streamlit file" to `streamlit_app.py`.
4. Click "Deploy". The app will install packages from requirements.txt and apt packages from apt.txt.

Notes:
- If you see import errors related to `libGL.so.1` or similar, apt.txt installs the missing system packages on Streamlit Cloud. For other hosts (Docker), install these system packages in the image (see Docker example in main instructions).
- Streamlit Cloud redeploys automatically when you push to the selected branch.

Local testing:
- Create & activate a virtualenv, then:
  ```
  pip install -r requirements.txt
  streamlit run streamlit_app.py
  ```

Docker (optional):
- Use a Dockerfile that installs the system packages before `pip install`.
```