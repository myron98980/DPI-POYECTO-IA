from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_webcam(conf, model, is_display_tracker, tracker):
    st.write("Entró a play_webcam")
    source_webcam = settings.WEBCAM_PATH

    try:
        vid_cap = cv2.VideoCapture(source_webcam)
        if not vid_cap.isOpened():
            st.write("Error al abrir la cámara web.")

        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.sidebar.error("Error loading webcam: " + str(e))


def play_youtube_video(conf, model, is_display_tracker, tracker):
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model, is_display_tracker, tracker):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')

    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_stored_video(conf, model, is_display_tracker, tracker):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def main():
    model_path = settings.DETECTION_MODEL
    model = load_model(model_path)
    conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    is_display_tracker, tracker = display_tracker_options()

    source_option = st.sidebar.selectbox("Select Source", settings.SOURCES_LIST)

    if source_option == settings.WEBCAM:
        play_webcam(conf, model, is_display_tracker, tracker)
    elif source_option == settings.YOUTUBE:
        play_youtube_video(conf, model, is_display_tracker, tracker)
    elif source_option == settings.RTSP:
        play_rtsp_stream(conf, model, is_display_tracker, tracker)
    elif source_option == settings.VIDEO:
        play_stored_video(conf, model, is_display_tracker, tracker)


if __name__ == "__main__":
    main()

