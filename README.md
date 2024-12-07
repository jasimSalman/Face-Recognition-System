# Installation

Prerequisites: Ensure you have Python 3 (version 3.3 or later) installed on your system. You can verify this by running python3 --version in your terminal. If not installed, download it from https://www.python.org/downloads/.

## Package Installation

To install the required Python libraries:

1. Open a terminal or command prompt.
2. Navigate to the directory containing your project's `requirements.txt` file.
3. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

This command will download and install all the packages listed in the `requirements.txt` file, ensuring your application has the necessary dependencies.

---

## Running the Application

1. Open a terminal or command prompt.
2. Navigate to the directory containing your `app.py` file.
3. Start the application by running:

   ```bash
   python3 app.py
   ```

## Using the Application

The application offers **two options** for face recognition:

---

### 1. Real-Time Recognition

- **Name Entry**:  
  Upon launch, you'll be prompted to enter your name. This name will be associated with the detected face and used for training the model.

- **Start Detection**:  
  Click the **"Start Detection"** button. The application will access your webcam and start capturing video frames. It will attempt to detect faces in the video stream and train the model.

- **Recognize**:  
  Click the **"Recognize"** button to begin real-time face recognition. The application will compare detected faces to the trained model and identify them if possible.

---

### 2. Face Image Recognition

- **Image Selection**:  
  Click the **"Select Images"** button (or equivalent functionality in the application). This will open a file dialog box, allowing you to select one or more images for face recognition.

- **Start Recognition**:  
  Click the **"Start Recognition"** button to initiate the recognition process on the selected images. The application will analyze the images to detect and identify faces, potentially matching them against a pre-trained model (if applicable).
