import streamlit as st
import yolo
import tempfile
import os

# Daftar model YOLO yang tersedia
MODEL_OPTIONS = {
                 "YOLOv8n": "best.pt"
                }

# Daftar sample video (bisa ditambahkan lebih banyak)
VIDEO_OPTIONS = {
                "Sample Video 1": "media/sample1.mp4",
                "Sample Video 2": "media/sample2.mp4"
                }

MARKING_PARAM = {
                "sample1": [1094, 231, 1083, 403, 15, 15, 10],
                "sample2": [423, 167, 423, 313, 15, 15, 10]
                }

def main():
    st.title("Realtime Water Level Detection")

    # Layout menggunakan columns
    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox("Select Model:", list(MODEL_OPTIONS.keys()))

    with col2:
        selected_video = st.selectbox("Select Video Source:", list(VIDEO_OPTIONS.keys()))

    video_path = VIDEO_OPTIONS[selected_video]

    # Jika pilih "Upload Your Own", tampilkan file uploader
    if selected_video == "Upload Your Own":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

    if video_path and st.button("Start Processing"):
        stframe = st.empty()

        # Jalankan YOLO dengan model dan video yang dipilih
        if selected_video == "Sample Video 1":
            for frame in yolo.yolo(video_path, MODEL_OPTIONS[selected_model], 1094, 231, 1083, 403, 15, 15, 10):
                stframe.image(frame, channels="BGR", use_column_width=True)
        
        if selected_video == "Sample Video 2":
            st.write('Unavaiable')
            # for frame in yolo.yolo(video_path, MODEL_OPTIONS[selected_model], 423, 167, 423, 313, 15, 15, 10):
            #     stframe.image(frame, channels="BGR", use_column_width=True)
        

if __name__ == "__main__":
    main()
