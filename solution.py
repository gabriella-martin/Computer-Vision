
import cv2
import time
from keras.models import load_model
import numpy as np
import random

computer_wins = 0
user_wins = 0


model = load_model('keras_model.h5')

def play_game():
   
    user_prediction = get_image()
    computer_choice = get_computer_choice()
    user_choice = get_prediction(user_prediction)
    get_winner(computer_choice, user_choice)
    check_score()

def get_image():

    start_time = time.time()

    while True: 
        cap = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        duration = 3
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)
        # Press q to close the window
        prediction = (np.argmax(prediction[0]))
        current_time = time.time()
        time_elapsed = current_time - start_time
        if cv2.waitKey(1) & 0xFF == ord('q') or time_elapsed > duration:
            break

    return prediction  
def check_score():
    if computer_wins == 3:
        print('Computer was the first to win 3 times, you lose!')
    elif user_wins == 3:
        print('You are the first to win 3, congrats')
    else:
        time.sleep(3)
        print('Try again!')
        play_game()

def get_prediction(user_prediction):
    if user_prediction == 0:
        user_choice = 'Rock'
      
    elif user_prediction == 1:
        user_choice = 'Paper'
        
    elif user_prediction == 2:
        user_choice = 'Scissors'
        
    else:
        user_choice = 'Nothing'
    
    return user_choice

def get_computer_choice():

    choices = ['Rock', 'Paper', 'Scissors']
    computer_choice = random.choice(choices)
    return computer_choice

def get_winner(computer_choice, user_choice):
    global computer_wins
    global user_wins
    # draws
    if computer_choice == user_choice:
        print(f"Computer also chose {computer_choice.lower()}, it's a draw")
       
    
    # losing scenarios
    elif (computer_choice == 'Rock' and user_choice == 'Scissors') or (computer_choice == 'Scissors' and user_choice == 'Paper') or (computer_choice == 'Paper' and user_choice == 'Rock'):
        print(f'Computer chose {computer_choice.lower()}, you chose {user_choice.lower()}, you lose!')
       
        computer_wins += 1
       
    elif user_choice == 'Nothing':
        print(f'You chose {user_choice.lower()} you lose!')
        computer_wins +=1
    
    #winning scenarios
    else:
        print(f'You chose {user_choice.lower()}, computer chose {computer_choice.lower()}, you win!')
        user_wins +=1

play_game()