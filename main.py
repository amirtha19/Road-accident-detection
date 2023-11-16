import cv2
import streamlit as st
import inference
import supervision as sv
from roboflow import Roboflow

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

st.title("Real-time Object Detection")

# Function to process uploaded video
def process_uploaded_video(uploaded_file):
    with st.spinner("Processing uploaded video..."):
        file_bytes = uploaded_file.read()
        # Perform object detection here using the code you provided
        # You might need to adapt the code for file-based input

# Sidebar option to choose input source
source_option = st.sidebar.radio("Select input source", ("Webcam", "Upload Video"))

if source_option == "Webcam":
    st.sidebar.write("Using webcam as the input source")
    st.sidebar.text("Ensure your webcam is connected.")
    inference.Stream(
        source="webcam",  # Use webcam as the source
        model="live-road-detection/6",  # Use the specified model
        output_channel_order="BGR",
        use_main_thread=True,  # For OpenCV display
        on_prediction=on_prediction,
    )
else:
    st.sidebar.write("Upload a video file for object detection")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

    if uploaded_file:
        process_uploaded_video(uploaded_file)
