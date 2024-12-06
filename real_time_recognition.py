import cv2
import os
import numpy as np
import json
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox


class RealtimeRecognition:
    def __init__(self, root):
        self.root = root
        self.real_time_frame = tk.Frame(root.root)

        self.face_info_saved = False
        self.face_id = None  
        self.face_name = None
        
        self.stop_recognition_button = None

        self.images_dir = './images/'
        self.cascade_classifier_filename = './trained_model/haarcascade_frontalface_default.xml'
        self.names_json_filename = 'names.json'
        self.trainer_filename = 'trainer.yml'
        self.create_directory(self.images_dir) ## Create image dictionary if not exist

        self.count = 0

        self.camera_label = None
        self.cap = None
        self.running = False

    def create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, font=("Helvetica", 14), bg="#4CAF50", fg="white", relief="flat", width=20, height=2, command=command)
        button.pack(pady=15)
        button.config(activebackground="#45a049", activeforeground="white")

    def create_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_face_id(self, directory: str) -> int:
        user_ids = []
        for filename in os.listdir(directory):
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
        user_ids = sorted(list(set(user_ids)))
        max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
        return max_user_ids

    def save_name(self,face_id: int, face_name: str, filename: str) -> None:
        names_json = {}
        if not os.path.exists(filename):
            with open(filename, 'w') as fs:
                json.dump(names_json, fs, ensure_ascii=False, indent=4)

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as fs:
                try:
                    names_json = json.load(fs)
                except json.JSONDecodeError:
                    print(f"Error: {filename} contains invalid JSON. Starting with an empty dictionary.")
                    names_json = {}

        names_json[str(face_id)] = face_name

        with open(filename, 'w') as fs:
            json.dump(names_json, fs, ensure_ascii=False, indent=4)

    def show_real_time(self):
        self.switch_frame(self.real_time_frame)
        self.setup_real_time_ui()

    def switch_frame(self, frame):
        if self.root.main_menu.winfo_ismapped():
            self.root.main_menu.pack_forget()
        frame.pack(fill="both", expand=True)    

    def setup_real_time_ui(self):
        if not self.camera_label:
            self.camera_label = tk.Label(self.real_time_frame)
            self.camera_label.pack()

        self.stop_recognition_button = tk.Button(self.real_time_frame, text = "Stop recognition", font= ("Helvetica", 14), bg="#4CAF50", fg="white", relief="flat", width=20, height=2, command= self.stop_recognition)
        self.stop_recognition_button.config(activebackground="#45a049", activeforeground="white")

        tk.Label(self.real_time_frame, text="Enter Name:", font=("Arial", 14)).pack(pady=10)

        self.name_entry = tk.Entry(self.real_time_frame, font=("Helvetica", 20), width=20 , bd=2, relief="solid")
        self.name_entry.pack(pady=15)

        self.create_button(self.real_time_frame, "Start Detection" ,self.start_camera)
        self.create_button(self.real_time_frame, "Recognize", self.start_recognition)
        self.create_button(self.real_time_frame, "Back to Menu", self.back_to_main_menu)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_camera()

    def back_to_main_menu(self):
        if self.cap:
            self.running = False
            self.cap.release()
            self.camera_label.config(image="")
            cv2.destroyAllWindows()

        if hasattr(self.root, 'switch_to_main_menu'):
            self.root.switch_to_main_menu()

    def update_camera(self):
        if not self.face_info_saved:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_classifier_filename)
            self.face_id = self.get_face_id(self.images_dir)
            self.face_name = self.name_entry.get()
            self.save_name(self.face_id, self.face_name, self.names_json_filename)
            self.face_info_saved = True 

        if self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,  minSize=(20, 20))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    self.count += 1
                    face_region = gray[y:y + h, x:x + w]
                    face_path = f'./images/Users-{self.face_id}-{self.count}.jpg'
                    cv2.imwrite(face_path, face_region)

                if self.count >= 30:
                    self.running = False
                    self.cap.release()
                    cv2.destroyAllWindows()
                    self.on_capture_complete()
                    return

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                
                if not self.camera_label:
                    self.camera_label = tk.Label(self.real_time_frame)
                    self.camera_label.pack(pady=10)
                self.camera_label.config(image=img)
                self.camera_label.image = img
            
            self.camera_label.after(10, self.update_camera)

    def on_capture_complete(self):
        messagebox.showinfo("Info", f"Captured {self.count} images successfully!")
        try:
            self.train_face_recognizer('./images/')
            messagebox.showinfo("Info", "Face recognizer training completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")
            
        self.stop_cam()

    def train_face_recognizer(self, path: str):
        print("\n[INFO] Training face recognizer...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self.cascade_classifier_filename)

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            face_samples = []
            ids = []
            for image_path in image_paths:
                PIL_img = Image.open(image_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                face_id = int(os.path.split(image_path)[-1].split("-")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(face_id)
            return face_samples, ids

        faces, ids = get_images_and_labels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write(self.trainer_filename)
        print(f"\n[INFO] {len(np.unique(ids))} faces trained.")

    def load_resources(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.trainer_filename)
        self.face_cascade = cv2.CascadeClassifier(self.cascade_classifier_filename)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        with open(self.names_json_filename, 'r') as fs:
            names = json.load(fs)
            self.names = names

    def start_recognition(self):

        recognition_thread = threading.Thread(target=self.recognize_real_time)
        recognition_thread.daemon = True  
        recognition_thread.start()

    def recognize_real_time(self):
        self.load_resources()
        
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not access the camera.")
            return

        if self.stop_recognition_button and not self.stop_recognition_button.winfo_ismapped():
            self.stop_recognition_button.pack(pady=15)

        while self.running:
            ret, img = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 50:
                    try:
                        name = self.names.get(str(id), "Unknown") 
                        confidence_text = f"{100 - round(confidence)}%"
                    except Exception as e:
                        name = "Unknown"
                        confidence_text = "N/A"
                else:
                    name = "Unknown"
                    confidence_text = "N/A"
                
                cv2.putText(img, name, (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 300))
            img = ImageTk.PhotoImage(Image.fromarray(img))


            if not self.camera_label:
                self.camera_label = tk.Label(self.real_time_frame)
                self.camera_label.pack(pady=10)
            self.camera_label.config(image=img)
            self.camera_label.image = img

            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False

    def stop_recognition(self):
        if self.cap:
            self.running = False
            self.cap.release()
            self.camera_label.config(image="")
            cv2.destroyAllWindows()
        
        if self.stop_recognition_button and self.stop_recognition_button.winfo_ismapped():
            self.stop_recognition_button.pack_forget()

    def  stop_cam(self):
        if self.cap:
            self.running = False
            self.cap.release()
            self.camera_label.config(image="")
            cv2.destroyAllWindows()