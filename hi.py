from ultralytics import YOLO
video = r"C:\Users\amirt\Downloads\327-1_327-2865_preview (1).mp4"
def process_video(video):
    model = YOLO("best.pt")
    results = model.predict(video,save=True)


process_video(video)