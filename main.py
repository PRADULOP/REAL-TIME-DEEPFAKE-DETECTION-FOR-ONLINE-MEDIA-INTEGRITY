import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import cv2
import torch
import torchvision.transforms as transforms
from tkinter import filedialog
from torchvision import models
import torch.nn as nn
import pyttsx3

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(x.size(0), -1)
        x_dp = self.dp(x)
        output = self.linear1(x_dp)
        return fmap, output

# Initialize the Tkinter window
window = tk.Tk()
window.title("Deepfake Detector")

# Set the desired window size
window_width = 800
window_height = 600
window.geometry(f"{window_width}x{window_height}")

# Load and resize the background image
background_image_path = r"/Users/pradulsmacbookair/Downloads/Projects/Mini Project/bg.png"  # Path to your background image
background_image = Image.open(background_image_path)
background_image = background_image.resize((window_width, window_height), Image.LANCZOS)

# Reduce opacity of the background image
enhancer = ImageEnhance.Brightness(background_image)
background_image = enhancer.enhance(0.5)  # Reduce opacity by 50%

# Convert PIL Image to Tkinter PhotoImage
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Initialize the text-to-speech engine
#engine = pyttsx3.init()

# Create a label for the video feed
video_label = tk.Label(window)
video_label.pack(pady=10)

# Create a label to display the prediction result
label = tk.Label(window, font=("Arial", 16))
label.pack(pady=10)

# Instantiate your custom Model (modify parameters as needed)
model = Model(num_classes=2)

# Set the path to the saved state dictionary file
model_path = r"/Users/pradulsmacbookair/Downloads/Projects/Mini Project/deepfakenet.pt"  # Path to your trained model file

# Load the state dictionary (ensure map_location='cpu' if not using CUDA)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Initialize the webcam and other global variables
cap = None
is_detecting = False

# Function to perform video prediction
def predict_video(video_path=None):
    global cap, is_detecting
    if video_path:
        cap = cv2.VideoCapture(video_path)
    if cap is None or not cap.isOpened():
        return

    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (224, 224))
    pil_image = Image.fromarray(frame_resized)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        _, output = model(input_tensor)

    predicted_class = torch.argmax(output, dim=1).item()

    if predicted_class == 0:
        result = "Not Deepfake"
    else:
        result = "Deepfake"

    if is_detecting:
        label.config(text=f"Prediction: {result}")
        speak_prediction(result)

        # Update the video feed
        imgtk = ImageTk.PhotoImage(image=pil_image)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if is_detecting:
            window.after(10, predict_video)

# Function to start webcam detection
def toggle_webcam_on():
    global cap, is_detecting
    if not is_detecting:  # Check if not already detecting
        cap = cv2.VideoCapture(1)  # Open webcam
        is_detecting = True
        predict_video()  # Start video prediction

# Function to stop webcam detection
def toggle_webcam_off():
    global cap, is_detecting
    is_detecting = False
    if cap is not None:
        cap.release()  # Release webcam
        cap = None

# Function to start video prediction
def start_detection():
    global is_detecting
    is_detecting = True
    upload_video()  # Use uploaded video for prediction

# Function to stop video prediction
def stop_detection():
    global is_detecting
    is_detecting = False

# Function to upload video file
def upload_video():
    video_path = filedialog.askopenfilename()
    if video_path:
        predict_video(video_path)

# Function to speak the prediction result
#def speak_prediction(result):
    #engine.say(f"The prediction is {result}")
    #engine.runAndWait()

# Create start and stop buttons with purple color
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

button_texts = ["Start Prediction", "Stop Prediction"]
button_commands = [start_detection, stop_detection]

for text, command in zip(button_texts, button_commands):
    button_color = '#800080'  # Purple color
    button = tk.Button(button_frame, text=text, command=command)
    button.pack(side=tk.LEFT, padx=0, expand=True, fill=tk.X)
    button.configure(bg=button_color, fg='white', font=("Arial", 14, "bold"))

# Create button to toggle webcam on/off
webcam_button = tk.Button(button_frame, text="Toggle Webcam", command=toggle_webcam_on)
webcam_button.pack(side=tk.LEFT, padx=0, expand=True, fill=tk.X)
webcam_button.configure(bg='#800080', fg='white', font=("Arial", 14, "bold"))

# Apply a stylish template and design
window.configure(bg='#F0F0F0')  # Set background color

# Run the Tkinter event loop
window.mainloop()

# Release the webcam or video capture when done
if cap is not None:
    cap.release()
 