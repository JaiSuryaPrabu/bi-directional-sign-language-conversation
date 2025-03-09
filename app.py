import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
from huggingface_hub import spaces

# Define the ASLClassifier model
class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, num_classes=28):
        super(ASLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn4 = nn.BatchNorm1d(hidden_size // 2)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        return x

# Load the model and label encoder (CPU initially, GPU handled by decorator)
device = torch.device('cpu')  # Default to CPU; GPU inference handled by @spaces.GPU
model = ASLClassifier().to(device)
model.load_state_dict(torch.load('data/asl_classifier.pth', map_location=device))
model.eval()

df = pd.read_csv('data/asl_landmarks_final.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['label'].values)

# Initialize MediaPipe (runs on CPU)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Prediction function with GPU offloading
@spaces.GPU
def predict_letter(landmarks, model, label_encoder):
    with torch.no_grad():
        # Move to GPU for inference (handled by decorator)
        landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to('cuda')
        model = model.to('cuda')
        output = model(landmarks)
        _, predicted_idx = torch.max(output, 1)
        letter = label_encoder.inverse_transform([predicted_idx.item()])[0]
        # Move model back to CPU to free GPU memory
        model = model.to('cpu')
    return letter

# Video processing function (CPU for video processing, GPU for prediction)
def process_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."

    # Variables to store output
    text_output = ""
    out_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe (CPU)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks and predict (GPU via decorator)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks, dtype=np.float32)
                predicted_letter = predict_letter(landmarks, model, label_encoder)

                # Add letter to text (avoid duplicates if same as last)
                if not text_output or predicted_letter != text_output[-1]:
                    text_output += predicted_letter

                # Overlay predicted letter on frame
                cv2.putText(frame, f"Letter: {predicted_letter}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Store processed frame
        out_frames.append(frame)

    cap.release()

    # Write processed video to a temporary file
    out_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (out_frames[0].shape[1], out_frames[0].shape[0]))
    for frame in out_frames:
        out.write(frame)
    out.release()

    return out_path, text_output

# Create Gradio interface
with gr.Blocks(title="Sign Language Translation") as demo:
    gr.Markdown("## Sign Language Translation")
    video_input = gr.Video(label="Input Video", sources=["upload", "webcam"])
    video_output = gr.Video(label="Processed Video with Landmarks")
    text_output = gr.Textbox(label="Predicted Text", interactive=False)

    # Button to process video
    btn = gr.Button("Translate")
    btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[video_output, text_output]
    )

# Launch the app
demo.launch()