import pickle
from collections import Counter
from pathlib import Path
import face_recognition
from PIL import Image, ImageDraw, ImageFont ,ImageTk
import tkinter as tk
from tkinter import filedialog
import threading
import sys

class FaceRecognition:
    def __init__(self, root):
        self.root = root
        self.face_recognition_frame = tk.Frame(root.root)

        self.ready_encoding_path = Path('./trained_model/trained_model.pkl')
        
        self.rec_position_color = "green"
        self.text_color = "white"
        
        self.files = []
        self.pre_names = []
        self.images = []
    
    def predict_images(self, model="hog"):
        for filepath in self.files:
            self.prediction(filepath, model=model)

    def prediction(self, image, model="hog"):
        print("start prediction ")
        with self.ready_encoding_path.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
            if loaded_encodings:
                print("loading the model")

        input_image = face_recognition.load_image_file(image)
        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for rec_position, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name = self.recognition(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            self.draw_face(draw, rec_position, name)
            self.pre_names.append(name)
            self.images.append(pillow_image.copy())
        self.show_predicted_images(pillow_image, image, name)
    
    def show_predicted_images(self , image,input_image_path ,name):

        input_image = Image.open(input_image_path) 
        input_image.thumbnail((400, 400))  
        input_photo = ImageTk.PhotoImage(input_image)

        image.thumbnail((400, 400))
        predicted_photo = ImageTk.PhotoImage(image)

        input_frame = tk.Frame(self.image_container, bg="#F0F0F0")
        input_frame.pack(side=tk.LEFT, padx=5, pady=5)

        input_lbl = tk.Label(input_frame, image=input_photo, bg="#F0F0F0")
        input_lbl.image = input_photo  
        input_lbl.pack()

        input_name_lbl = tk.Label(input_frame, text="Input Image", bg="#F0F0F0", font=("Helvetica", 12))
        input_name_lbl.pack()

        predicted_frame = tk.Frame(self.image_container, bg="#F0F0F0")
        predicted_frame.pack(side=tk.LEFT, padx=5, pady=5)

        predicted_lbl = tk.Label(predicted_frame, image=predicted_photo, bg="#F0F0F0")
        predicted_lbl.image = predicted_photo  
        predicted_lbl.pack()

        predicted_name_lbl = tk.Label(predicted_frame, text=name, bg="#F0F0F0", font=("Helvetica", 12))
        predicted_name_lbl.pack()
        
    def recognition(self, unknown_encoding, loaded_encodings):
        boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
        selected = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
        if selected:
            return selected.most_common(1)[0][0]

    def draw_face(self, draw, rec_position, name):
        top, right, bottom, left = rec_position

        draw.rectangle(((left, top), (right, bottom)), outline=self.rec_position_color, width=5)

        box_height = bottom - top
        font_size = max(20, box_height // 4) 
        try:
            font = ImageFont.truetype("Helvetica", font_size)  
        except IOError:
            font = ImageFont.load_default()  

        text_bbox = font.getbbox(name)  
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_left, text_top = left, bottom
        text_right, text_bottom = text_left + text_width + 10, text_top + text_height + 10

        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=self.rec_position_color, outline=self.rec_position_color)

        draw.text((text_left + 5, text_top + 5), name, fill=self.text_color, font=font)

    def setup_image_recognition_ui(self):
        self.root.create_button(self.face_recognition_frame, "Upload Images", self.upload_images)
        self.image_container = tk.Frame(self.face_recognition_frame, bg="#F0F0F0")
        self.image_container.pack(pady=20)
        self.root.create_button(self.face_recognition_frame, "Start Recognition", self.recognize_images)
        self.root.create_button(self.face_recognition_frame, "Back to Menu", self.back_to_main_menu)

    def recognize_images(self):
        self.pre_names.clear()
        self.images.clear()
        for widget in self.image_container.winfo_children():
            widget.destroy()
        threading.Thread(target=self.predict_images).start()
    
    def upload_images(self):
        self.files.clear()
        for widget in self.image_container.winfo_children():
            widget.destroy()

        filepaths = filedialog.askopenfilenames(multiple=True)
        for filepath in filepaths:
            self.files.append(filepath)
            img = Image.open(filepath)
            img.thumbnail((100, 100))  
            img = img.convert("RGB")  
            
            photo = ImageTk.PhotoImage(img)
            
            lbl = tk.Label(self.image_container, image=photo, bg="#F0F0F0")
            lbl.image = photo  
            lbl.pack(side=tk.TOP, padx=5, pady=5)
            
    def switch_frame(self, frame):
        if self.root.main_menu and self.root.main_menu.winfo_ismapped():
            self.root.main_menu.pack_forget()
        frame.pack(fill="both", expand=True)   

    def show_face_recognition(self):
        self.switch_frame(self.face_recognition_frame)
        self.setup_image_recognition_ui()
        

    def back_to_main_menu(self):
        self.files = []
        self.pre_names = []
        self.images = []
        if hasattr(self.root, 'switch_to_main_menu'):
            self.root.switch_to_main_menu()