import av  # noqa: F401 (import needed by streamlit-webrtc even if not referenced directly)
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image
from utils import Segmenter, FaceFinder, apply_background, crop_to_face
import tempfile
import os

st.set_page_config(page_title="Face Focus & Background Replace", page_icon="🎥", layout="wide")

st.title("🎥 Face Focus & Background Replace (Webcam/Video)")
st.caption("OpenCV + MediaPipe | Live webcam via WebRTC or process an uploaded video")

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Input Source", ["Webcam (Live)", "Upload Video"])
    face_only = st.checkbox("Show only the main face (crop)", value=False)
    bg_mode = st.selectbox("Background Mode", ["blur", "replace", "remove"], index=0)
    blur_ksize = st.slider("Blur strength (ksize)", min_value=11, max_value=99, value=35, step=2)
    seg_thresh = st.slider("Segmentation threshold", 0.1, 0.9, 0.6, 0.05)

    bg_image = None
    if bg_mode == "replace":
        bg_file = st.file_uploader("Upload replacement background (image)", type=["png", "jpg", "jpeg"])
        if bg_file:
            bg_image = Image.open(bg_file).convert("RGB")
            bg_image = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)

st.markdown("---")

segmenter = Segmenter(model_selection=1)
facefinder = FaceFinder(model_selection=1, min_detection_confidence=0.6)

def process_frame(frame_bgr):
    # Optional face crop
    if face_only:
        bbox = facefinder.largest_face_bbox(frame_bgr)
        frame_bgr = crop_to_face(frame_bgr, bbox)

    # Segmentation
    m = segmenter.mask(frame_bgr, threshold=seg_thresh)

    # Background effect
    if bg_mode == "replace":
        out = apply_background(frame_bgr, m, mode="replace", bg_bgr=bg_image)
    elif bg_mode == "remove":
        out = apply_background(frame_bgr, m, mode="remove")
    else:
        out = apply_background(frame_bgr, m, mode="blur", blur_ksize=blur_ksize)
    return out

if mode == "Webcam (Live)":
    st.subheader("Live Webcam")
    st.info("Allow camera access in your browser. Processing runs on your machine (no upload).")

    class Transformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            out = process_frame(img)

            # streamlit-webrtc expects BGR24 (no alpha). If remove mode, convert from BGRA.
            if out.ndim == 3 and out.shape[2] == 4:
                out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
            return out

    webrtc_streamer(
        key="demo",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

else:
    st.subheader("Process Uploaded Video")
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    col1, col2 = st.columns(2)
    with col1:
        preview = st.empty()
    with col2:
        st.write("Output will appear below after processing.")

    if uploaded is not None:
        # Use cross-platform temp directory
        temp_dir = tempfile.gettempdir()
        tfile = os.path.join(temp_dir, uploaded.name)

        with open(tfile, 'wb') as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tfile)
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(temp_dir, "processed.mp4")

            writer = None
            st.write("Processing... this runs locally in your browser session.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                out = process_frame(frame)

                # Show preview frame
                if out.ndim == 3 and out.shape[2] == 4:
                    # Convert BGRA to BGR for preview
                    show = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
                else:
                    show = out

                # Initialize writer lazily after we know shape
                if writer is None:
                    h, w = show.shape[:2]
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                writer.write(show)

                # Stream preview (downsample for speed)
                pr_w = min(640, show.shape[1])
                pr_h = int(pr_w * show.shape[0] / show.shape[1])
                pr = cv2.resize(show, (pr_w, pr_h))
                preview.image(cv2.cvtColor(pr, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            if writer:
                writer.release()

            st.success("Done! Download the processed video:")
            with open(out_path, "rb") as f:
                st.download_button("Download processed.mp4", data=f, file_name="processed.mp4", mime="video/mp4")

            if bg_mode == "remove":
                st.info("You chose 'remove' mode. MP4 cannot store transparency. If you need transparent frames, export PNGs.")
                # Export the first frame as transparent PNG
                cap2 = cv2.VideoCapture(tfile)
                ret, first = cap2.read()
                if ret:
                    m = segmenter.mask(first, threshold=seg_thresh)
                    bgra = apply_background(first, m, mode="remove")
                    png_path = os.path.join(temp_dir, "frame0_transparent.png")
                    cv2.imwrite(png_path, bgra)
                    with open(png_path, "rb") as f:
                        st.download_button("Download first frame (transparent PNG)", data=f, file_name="frame0_transparent.png", mime="image/png")
                cap2.release()