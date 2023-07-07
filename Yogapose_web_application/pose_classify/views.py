
# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render

import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
# from . import yogaposeapp
from django.conf import settings

def index(request):
    return render(request,'templates/index.html')

from collections import Counter

def predict_action(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        file_path = fs.save(video_file.name, video_file)
        video_path = fs.path(file_path)
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yoga_pose.h5')
        LRCN_model = load_model(model_path)


        # Specify the height and width to which each video frame will be resized in our dataset.
        IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

        # Specify the number of frames of a video that will be fed to the model as one sequence.
        SEQUENCE_LENGTH = 150
        CLASSES_LIST = ['Bhujangasana', 'Padmasana', 'Shavasana','Tadasana','Trikonasana','Vrikshasana']  # Replace with your own class names

        output_video_path = os.path.join(settings.MEDIA_ROOT, 'output.mp4')


        output_list = []
        predict_on_video(video_path, output_video_path, SEQUENCE_LENGTH, LRCN_model, CLASSES_LIST, output_list,IMAGE_HEIGHT,IMAGE_WIDTH)
        frequency_counter = Counter(output_list)
        predicted_output = max(frequency_counter, key=frequency_counter.get)
        print("Predicted Class is ", predicted_output)
        

        return render(request,'output.html',{'predicted_output':predicted_output})
    
        # return JsonResponse({'predictions': output_list})
    return render(request,'templates/predict_action.html')

def predict_on_video(video_path,output_video_path,SEQUENCE_LENGTH,LRCN_model,CLASSES_LIST,output_list,IMAGE_HEIGHT,IMAGE_WIDTH):
    video_reader=cv2.VideoCapture(video_path)
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    predicted_class_name = ''
    while video_reader.isOpened():
        ok, frame = video_reader.read() 
 
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:

            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            
            # print("Predicted Class is ", predicted_class_name)

        output_list.append(predicted_class_name)
        # cv2.putText(frame, predicted_class_name, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
    video_reader.release()


def about(request):
    return HttpResponse("About Harry Bhai")