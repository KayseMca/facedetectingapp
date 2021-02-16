import os
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

def facecrop(filename):  
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]
        
    # file_data = image.read()
    print("herer#########")
    # path = file_dir+'/static/uploads/'+filename
    path = 'static/uploads/'+str(filename)
    print(path)
    # facedata = "haarcascade_frontalface_default.xml"
    facedata = file_dir+'/haarcascade_frontalface_default.xml'
    # cascade = cv2.CascadeClassifier(cv2.data.haarcascades +facedata)
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(path)
    
    #converting to gray
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # reducing image size
    scaleFactor = 1.3

    # 5 neighbors should be present for each rectangle to be retained.
    minNeighbors = 5
    # cv2.imshow("image", img)
    # try:
    # minisize = (img.shape[1],img.shape[0])
    # # print(minisize)
    # miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(gray_frame, scaleFactor, minNeighbors)
    # faces = cascade.detectMultiScale(miniframe)

    if len(faces) ==0:
        return [img]

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        sub_face = gray_frame[y:y+h, x:x+w]        
        sub_face = cv2.resize(sub_face,(48,48))
        cv2.imwrite('static/uploads/cropped'+str(filename), sub_face)
        cv2.waitKey(0)
        
        # saving circled face
        xc = int((x + x+w)/2)
        yc = int((y + y+h)/2)
        radius = int(w/2)
        Thickness = 2

        circle = {"x":xc,"y":yc,"radius":radius}
        # Drawing the Circle on the Image
        # cv2.circle(img, (xc, yc), radius, (0, 255, 0))
        # path = file_dir+"/static/uploads/" + str(img)
        cv2.imwrite('static/uploads/detected'+str(filename), img)
        cv2.waitKey(0)
        


        # predicting the image
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            print("loading..")
            loading = True
            json_file = open(file_dir+"\module\model.json",'r')
            load_json_model = json_file.read()
            json_file.close()

            loaded_model = tf.keras.models.model_from_json(load_json_model)
            loaded_model.load_weights(file_dir+"\module\model_weights.h5")
            print("loaded...")
            loading = False

            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            output = loaded_model.predict(sub_face[np.newaxis, :, :, np.newaxis])

            pred = EMOTIONS_LIST[np.argmax(output)]

            # drawing the grapth
            data = output.tolist()[0]
            # Initializing the Figure for Bar Graph
            fig = plt.figure(figsize=(8, 5))
            objects = ('Angry', 'Discus', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')
            y_pos = np.arange(len(objects))
            plt.bar(y_pos, data, align='center', color='green', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xlabel("Emotions")
            plt.ylabel('Probability')
            plt.title('Facial Expression Recognation')
            # Saving the Bar Plot
            path = file_dir+"/static/uploads/graph"+str(filename)
            plt.savefig(path)

            return loading,circle,pred
