import pickle
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import pyttsx3

global char,c
bg_colour = '#121212'

# MAIN WINDOW
window = Tk()
window.geometry('1600x1024')
window.configure(bg=bg_colour)

# SECOND WINDOW
def asl():
    bg_colour = '#121212'

# BUTTON FUNCTIONS
def Add():
    c = Text.get()
    Text.delete(0, END)
    Text.insert(0, c + char)


def space():
    char = " "
    c = Text.get()
    Text.delete(0, END)
    Text.insert(0, c + char)
def clear():
    Text.delete(len(Text.get())-1, END)
def delete():
    Text.delete(0, END)
def voice():
    converter.say(Text.get())
    converter.runAndWait()


# IMAGES
Add_img = ImageTk.PhotoImage(Image.open('add.png'))
space_img = ImageTk.PhotoImage(Image.open('space.png'))
clear_img = ImageTk.PhotoImage(Image.open('clear.png'))
delete_img = ImageTk.PhotoImage(Image.open('delete.png'))
voice_img = ImageTk.PhotoImage(Image.open('voice.png'))
webcam_bg_img = ImageTk.PhotoImage(Image.open('web_bg.png'))
text_bg_img = ImageTk.PhotoImage(Image.open('Text.png'))
character_bg_img = ImageTk.PhotoImage(Image.open('character.png'))

# LABELS
webcam_bg = Label(window, borderwidth='0', image=webcam_bg_img)
webcam_bg.place(x=20,y=18)
webcam = Label(window,borderwidth='0')
webcam.place(x=40,y=38)
character_bg = Label(window, borderwidth='0', image=character_bg_img)
character_bg.place(x=770,y=129)
character = Label(window, borderwidth='0',background='#746969',font=('bold', 45))
character.place(x=1081,y=139)
text_bg = Label(window, borderwidth='0', image=text_bg_img)
text_bg.place(x=770,y=392)

# BUTTONS
Add_button = Button(window, image=Add_img, background=bg_colour,borderwidth='0',activeforeground=bg_colour,activebackground=bg_colour,command=Add)
Add_button.place(x=851, y=261)
space_button = Button(window, image=space_img, background=bg_colour, borderwidth='0',activeforeground=bg_colour,activebackground=bg_colour, command=space)
space_button.place(x=1253, y=538)
clear_button = Button(window, image=clear_img, background=bg_colour, borderwidth='0',activeforeground=bg_colour,activebackground=bg_colour, command=clear)
clear_button.place(x=851, y=538)
delete_button = Button(window, image=delete_img, background=bg_colour, borderwidth='0',activeforeground=bg_colour,activebackground=bg_colour, command=delete)
delete_button.place(x=1052, y=538)
voice_button = Button(window, image=voice_img, background=bg_colour, borderwidth='0',activeforeground=bg_colour,activebackground=bg_colour, command=voice)
voice_button.place(x=1042, y=261)

# ENTRY BOX
Text = Entry(window, background='#746969',borderwidth='0', font=('bold', 45))
Text.place(x=920, y=392, height=93, width=472)


# MAIN CODE

converter = pyttsx3.init()
converter.setProperty('rate', 150)
converter.setProperty('volume', 0.7)
model_dict = pickle.load(open('./one_hand_gestures.p', 'rb'))
model = model_dict['model']
model_dict1 = pickle.load(open('./one_hand_gestures.p', 'rb'))
model1 = model_dict1['model']

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict1 = {0: 'A', 1: 'B', 2: 'C',3: 'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z' }
while True:
    ret, img1 = cap.read()
    frame_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray((frame_rgb)))
    hand, img2 = detector.findHands(img1)
    try:
        if len(hand) == 2:
            data_aux = []
            x_ = []
            y_ = []
            webcam['image'] = img
            window.update()
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
               for hand_landmarks in results.multi_hand_landmarks:
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
               prediction = model1.predict([np.asarray(data_aux)])
               predicted_character = labels_dict1[int(prediction[0])]
               character['text'] = predicted_character
               char = predicted_character
        if len(hand) == 1:
            data_aux = []
            x_ = []
            y_ = []
            webcam['image'] = img
            window.update()
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict1[int(prediction[0])]
                character['text'] = predicted_character
                char = predicted_character
        else:
            webcam['image'] = img
            window.update()
    except:
        webcam['image'] = img
        window.update()
cap.release()
cv2.destroyAllWindows()
