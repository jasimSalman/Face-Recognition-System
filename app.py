import tkinter as tk
from real_time_recognition import RealtimeRecognition
from face_recognition1 import FaceRecognition


class RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection and Recognition System")
        self.root.geometry("600x400")
        self.root.config(bg="#F0F0F0")

        self.realtime_recognition = RealtimeRecognition(self)
        self.face_recognition = FaceRecognition(self)

        self.main_menu = None
        self.create_main_menu()

    def create_main_menu(self):
        self.main_menu = tk.Frame(root, bg="#F0F0F0")
        self.main_menu.pack(fill="both", expand=True)
        tk.Label(self.main_menu, text="Face Detection and Recognition", font=("Helvetica", 24, "bold"), bg="#F0F0F0", fg="#333").pack(pady=20)
        self.create_button(self.main_menu, "Real-Time Recognition", self.show_real_time)
        self.create_button(self.main_menu, "Image Recognition", self.show_face_recognition)

    def show_real_time(self):
        self.realtime_recognition.show_real_time()
    
    def show_face_recognition(self):
        self.face_recognition.show_face_recognition()

    def switch_to_main_menu(self):
        for widget in self.realtime_recognition.real_time_frame.winfo_children():
            widget.destroy()
        for widget in self.face_recognition.face_recognition_frame.winfo_children():
            widget.destroy()

        self.realtime_recognition.real_time_frame.pack_forget()
        self.face_recognition.face_recognition_frame.pack_forget()

        self.create_main_menu()

    def create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, font=("Helvetica", 14), bg="#4CAF50", fg="white", relief="flat", width=20, height=2, command=command)
        button.pack(pady=15)
        button.config(activebackground="#45a049", activeforeground="white")


if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    root.mainloop()
