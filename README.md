# Face Focus & Background Replace (Webcam/Video) — Streamlit + MediaPipe

This app lets you:
- Use **webcam** (live) or **upload a video**.
- **Show only the main face** (crop to the largest face) _optional_.
- **Remove, blur, or replace** background with a chosen image.
- Save processed output to a video file (for uploads) or just view live (for webcam).

## Quickstart

```bash
# 1) Create venv (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the Streamlit URL it prints (usually http://localhost:8501).

## Notes
- **Webcam (Live)**: Uses `streamlit-webrtc` for real-time processing.
- **Uploaded Video**: Processes the file and lets you download the result.
- **Background modes**: `remove` (transparent for PNG image downloads), `blur`, or `replace` (with your uploaded background image).
- **Face-only**: Crops video to the largest detected face with margin.

If you run into issues with webcam permissions, allow camera access in your browser and OS.
