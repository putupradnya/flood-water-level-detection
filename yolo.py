import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
import math
import numpy as np
import os
import requests
import time
from dotenv import load_dotenv


def load_font(size=45):
    """Load custom font for drawing text on image."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)


def send_telegram_alert(distance, warningLevel, image):
    """Send an alert message with an image to Telegram."""
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    message = f"⚠️ ALERT! Ketinggian air mencapai {distance:.2f} meter, melebihi batas {warningLevel} meter!"
    url_message = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    requests.get(url_message, params={"chat_id": CHAT_ID, "text": message})

    img_bytes = cv2.imencode(".jpg", np.array(image))[1].tobytes()
    requests.post(url_photo, data={"chat_id": CHAT_ID}, files={"photo": img_bytes})


def draw_dashed_line(draw, start, end, color, width=3, dash_length=15):
    """Draw a dashed line from start to end coordinates."""
    x1, y1 = start
    x2, y2 = end
    total_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_dashes = int(total_length / (2 * dash_length))
    
    for i in range(num_dashes):
        start_x = int(x1 + (x2 - x1) * (2 * i) / (2 * num_dashes))
        start_y = int(y1 + (y2 - y1) * (2 * i) / (2 * num_dashes))
        end_x = int(x1 + (x2 - x1) * (2 * i + 1) / (2 * num_dashes))
        end_y = int(y1 + (y2 - y1) * (2 * i + 1) / (2 * num_dashes))
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)

def draw_percentage_markers(draw, start, end, color, width=3, interval=0.2):
    """Draw percentage markers along the line with correct order (100% at bottom, 0% at top)."""
    x1, y1 = start
    x2, y2 = end
    for i in range(1, 5):  # 20%, 40%, 60%, 80%
        fraction = i * interval
        px = int(x1 + fraction * (x2 - x1))
        py = int(y1 + fraction * (y2 - y1))
        draw.line([(px - 5, py), (px + 5, py)], fill=color, width=width)
        draw.text((px + 10, py - 10), f"{int((1 - fraction) * 100)}%", fill=color, font=load_font(20))


def yolo(video_path, model_path, first_x, first_y, second_x, second_y, pixelsInAMeter, tipHeight, warningLevel):
    """YOLO model processing and real-time inference on video stream."""
    model = YOLO(model_path)
    myFont = load_font(50)
    last_alert_time = None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    def find_intersection(annotation, line):
        intersection = annotation.intersection(line)
        return [(intersection.xy)] if not intersection.is_empty else None

    def calculateDistance(x1, y1, x2, y2):
        return float("{:.2f}".format(tipHeight - (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / pixelsInAMeter))

    video_placeholder = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.4)
        segments = getattr(getattr(results[0], 'masks'), 'segments')[0]
        segmentsSize = int(segments.size / 2)
        segment = segments[0:segmentsSize]
        polygon_vertices = [(int(float(segment[i][0]) * frame.shape[1]), int(float(segment[i][1]) * frame.shape[0])) for i in range(segmentsSize)]

        line_coords = [(first_x, first_y), (second_x, second_y)]
        annotated_frame = results[0].plot()
        color_coverted = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        draw = ImageDraw.Draw(pil_image)

        # Draw reference line (blue)
        draw.line(line_coords, fill=(0, 0, 255), width=3)

        # 🔹 Hitung titik dashed line sejajar dengan reference line
        dx = second_x - first_x
        dy = second_y - first_y
        line_length = math.sqrt(dx**2 + dy**2)
        unit_dx = dx / line_length
        unit_dy = dy / line_length

        warning_x1 = int(first_x + unit_dx * (warningLevel * pixelsInAMeter))
        warning_y1 = int(first_y + unit_dy * (warningLevel * pixelsInAMeter))
        warning_x2 = int(second_x + unit_dx * (warningLevel * pixelsInAMeter))
        warning_y2 = int(second_y + unit_dy * (warningLevel * pixelsInAMeter))

        # 🔥 Gambar dashed warning line (merah, sejajar dengan reference)
        # draw_dashed_line(draw, (warning_x1, warning_y1), (warning_x2, warning_y2), color=(255, 0, 0), width=3)

        line = LineString(line_coords)

        if find_intersection(Polygon(polygon_vertices), line):
            intersection_points = find_intersection(Polygon(polygon_vertices), line)
            intersection_x = intersection_points[0][0][0]
            intersection_y = intersection_points[0][1][0]
            distance = calculateDistance(intersection_x, intersection_y, first_x, first_y)

            updated_line_coords = [(first_x, first_y), (intersection_x, intersection_y)]
            draw.line(updated_line_coords, fill=(0, 255, 0), width=3)  # Green line
            draw.text((1013, 134), str(distance), font=myFont, fill=(0, 0, 0))

            if distance >= warningLevel:
                draw.text((936, 98), "WARNING!!!", font=myFont, fill=(255, 0, 0))

                current_time = time.time()
                if last_alert_time is None or (current_time - last_alert_time) >= 10:
                    send_telegram_alert(distance, warningLevel, pil_image)
                    last_alert_time = current_time

        else:
            draw.text((936, 98), "SAFE", font=myFont, fill=(0, 255, 0))

        draw_percentage_markers(draw, (first_x, first_y), (second_x, second_y), color=(255, 255, 255), width=3)
        result = np.array(pil_image)
        video_placeholder.image(result, channels="RGB")

    cap.release()
    st.success("Processing complete!")
