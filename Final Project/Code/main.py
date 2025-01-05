import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import StringVar, Label, Frame, Button
from PIL import Image, ImageTk
import warnings
import pyttsx3
import threading

warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ',
    37: '.'
}
expected_features = 42

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.setProperty('volume', 0.85)

# Speak alphabet in a separate thread
def speak_alphabet(alphabet):
    def tts_thread():
        engine.say(alphabet)
        engine.runAndWait()

    threading.Thread(target=tts_thread, daemon=True).start()

# GUI Setup
root = tk.Tk()
root.title("Hand Gesture Translator")
root.geometry("900x700")
root.configure(bg="#f3f4f6")
root.resizable(False, False)

# Current Alphabet
current_alphabet = StringVar(value="N/A")
previous_alphabet = None

# Title
title_label = Label(
    root, text="Hand Gesture Translator", font=("Verdana", 26, "bold"), 
    fg="#2c3e50", bg="#f3f4f6", padx=10, pady=10
)
title_label.pack(pady=10)

# Row Layout: Video Frame and Buttons
row_frame = Frame(root, bg="#f3f4f6")
row_frame.pack(pady=10, fill="x")

# Video Frame
video_frame = Frame(row_frame, bg="#ecf0f1", bd=2, relief="ridge", width=600, height=450)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

video_label = tk.Label(video_frame, bg="#bdc3c7")
video_label.pack(fill="both", expand=True)

# Buttons Frame
button_frame = Frame(row_frame, bg="#f3f4f6")
button_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Buttons
start_button = Button(button_frame, text="Start", font=("Verdana", 14), bg="#27ae60", fg="white", width=12)
start_button.pack(pady=10)

stop_button = Button(button_frame, text="Stop", font=("Verdana", 14), bg="#e74c3c", fg="white", width=12)
stop_button.pack(pady=10)

exit_button = Button(button_frame, text="Exit", font=("Verdana", 14), bg="#34495e", fg="white", width=12, command=root.quit)
exit_button.pack(pady=10)

# Add Word and Sentence Frames
word_frame = Frame(root, bg="#ecf0f1", bd=2, relief="ridge")
word_frame.pack(pady=10, padx=20, fill="x")

Label(word_frame, text="Current Word:", font=("Verdana", 16), fg="#34495e", bg="#ecf0f1").pack(pady=5)
current_word = StringVar(value="N/A")
current_word_label = Label(word_frame, textvariable=current_word, font=("Verdana", 24, "bold"), fg="#27ae60", bg="#ecf0f1")
current_word_label.pack()

sentence_frame = Frame(root, bg="#ecf0f1", bd=2, relief="ridge")
sentence_frame.pack(pady=10, padx=20, fill="x")

Label(sentence_frame, text="Sentence:", font=("Verdana", 16), fg="#34495e", bg="#ecf0f1").pack(pady=5)
current_sentence = StringVar(value="N/A")
current_sentence_label = Label(sentence_frame, textvariable=current_sentence, font=("Verdana", 16), fg="#2c3e50", bg="#ecf0f1")
current_sentence_label.pack()

# Alphabet Display
alphabet_frame = Frame(root, bg="#ecf0f1", bd=2, relief="ridge")
alphabet_frame.pack(pady=20, padx=20, fill="x")

Label(alphabet_frame, text="Detected Alphabet:", font=("Verdana", 18), fg="#34495e", bg="#ecf0f1").pack(pady=5)
detected_alphabet_label = Label(alphabet_frame, textvariable=current_alphabet, font=("Verdana", 36, "bold"), fg="#27ae60", bg="#ecf0f1")
detected_alphabet_label.pack(pady=5)

# Video Capture
cap = cv2.VideoCapture(0)


def update_text(predicted_character):
    global current_word, current_sentence

    if predicted_character == " ":  # Space signals the end of a word
        if current_word.get() != "N/A":
            current_sentence.set(current_sentence.get() + " " + current_word.get())
            current_word.set("")  # Reset the current word
    elif predicted_character == ".":
        current_sentence.set(current_sentence.get() + ".")  # End the sentence
        current_word.set("")  # Reset the current word
    else:
        current_word.set(current_word.get() + predicted_character)

    # Update sentence text dynamically
    if current_sentence.get() == "N/A":
        current_sentence.set(current_word.get())
# Process frames
def process_frame():
    global current_alphabet, previous_alphabet

    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            # Predict gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            current_alphabet.set(predicted_character)
            # Update the text
            update_text(predicted_character)

            # Speak the alphabet if it changes
            if previous_alphabet != predicted_character:
                previous_alphabet = predicted_character
                speak_alphabet(predicted_character)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # Update video feed in GUI
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

# Start video processing
process_frame()
root.mainloop()


