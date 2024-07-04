import os
import time
import cv2
import easyocr
import imutils
import mysql.connector
import numpy as np
import pytesseract
import re
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def create_table():
    conn = mysql.connector.connect(host="localhost", user="root", password="root", database="be_project")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS vehicles (vehicle_number TEXT, bike_image_path TEXT)")
    conn.commit()
    conn.close()

def is_valid_license_plate(plate_text):
    # Pattern for standard Indian license plates
    pattern = re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{2}\s?\d{4}$')
    return bool(pattern.match(plate_text))

def insert_record(vehicle_number, bike_image_path):
    if not is_valid_license_plate(vehicle_number):
        print(f"Invalid license plate format: {vehicle_number}")
        return False
    conn = mysql.connector.connect(host="localhost", user="root", password="root", database="be_project")
    c = conn.cursor()
    c.execute("INSERT INTO vehicles VALUES (%s, %s)", (vehicle_number, bike_image_path))
    conn.commit()
    conn.close()
    return True

def select_video_file():
    filename = filedialog.askopenfilename(title="Select Video File")
    entry_video_file.delete(0, tk.END)
    entry_video_file.insert(0, filename)

def start_processing():
    video_file = entry_video_file.get()
    if not video_file:
        messagebox.showerror("Error", "Please select a video file.")
        return

    video_capture = cv2.VideoCapture(video_file)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_bike_results = person_bike_model.predict(img)

        for r in person_bike_results:
            boxes = r.boxes
            for box in boxes:
                cls = box.cls
                if person_bike_model.names[int(cls)] == "Person_Bike":
                    x1, y1, x2, y2 = box.xyxy[0]
                    person_bike_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    helmet_results = helmet_model.predict(person_bike_image)

                    for hr in helmet_results:
                        h_boxes = hr.boxes
                        for h_box in h_boxes:
                            h_cls = h_box.cls
                            if not helmet_model.names[int(h_cls)] == "With Helmet":
                                number_plate_results = number_plate_model.predict(person_bike_image)
                                for nr in number_plate_results:
                                    n_boxes = nr.boxes
                                    for n_box in n_boxes:
                                        n_cls = n_box.cls
                                        if number_plate_model.names[int(n_cls)] == "Number_Plate":
                                            # Process for number plate detection and text extraction
                                            cropped_plate_image = person_bike_image[int(n_box.xyxy[1]):int(n_box.xyxy[3]), int(n_box.xyxy[0]):int(n_box.xyxy[2])]
                                            reader = easyocr.Reader(['en'])
                                            result = reader.readtext(cropped_plate_image)
                                            if result:
                                                text = result[0][-2]
                                                text = re.sub(r'[^\w\s]', '', text)
                                                if is_valid_license_plate(text):
                                                    image_file = str(int(time.time())) + ".jpg"
                                                    output_file = f"person_violation_{image_file}"
                                                    output_path = os.path.join(output_dir, output_file)
                                                    cv2.imwrite(output_path, person_bike_image)

                                                    create_table()
                                                    insert_record(text, output_path)
                                                    print("Number Plate Text:", text)
                                                else:
                                                    print(f"Invalid license plate format detected: {text}")
        print("Process completed.")

# Initialize models
person_bike_model = YOLO(r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\First Model to detect motorcyclists\best.pt")
helmet_model = YOLO(r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\Second Model to detect helmet\best.pt")
number_plate_model = YOLO(r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Results\Models\Third Model to detect number plate\best.pt")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
output_dir = r"C:\Users\saket\OneDrive\Desktop\BE Project Code\BE Project Code\BE Project Code\Output"

# GUI setup
root = tk.Tk()
root.title("SafeRideGuard")
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label_video_file = tk.Label(frame, text="Video File:")
label_video_file.grid(row=0, column=0, sticky="e")
entry_video_file = tk.Entry(frame, width=50)
entry_video_file.grid(row=0, column=1, padx=5, pady=5)
button_browse = tk.Button(frame, text="Browse", command=select_video_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)
button_start = tk.Button(frame, text="Start Processing", command=start_processing)
button_start.grid(row=1, columnspan=3, pady=10)

root.mainloop()
