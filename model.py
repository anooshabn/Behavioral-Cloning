import csv
import cv2
import numpy as np
import sklearn
#from scipy import ndimage

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout, Activation
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def generator(lines, batch_size=32):
    n_lines = len(lines)
    while 1:  # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, n_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_line in batch_lines:
                for i in range(3): #images of center, left and right sides of the road
                    current_path = 'data/IMG/' + batch_line[i].split('/')[-1]
                    current_image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                    images.append(current_image)
                    
                    center_measurement = float(batch_line[3])
                    if i == 0:
                        measurements.append(center_measurement)
                    elif i == 1: 
                        measurements.append(center_measurement + 0.4)
                    elif i == 2: 
                        measurements.append(center_measurement - 0.4)
                    
                    images.append(cv2.flip(current_image, 1))
                    if i == 0:
                        measurements.append(center_measurement * -1.0)
                    elif i == 1: 
                        measurements.append((center_measurement + 0.4) * -1.0)
                    elif i == 2: 
                        measurements.append((center_measurement - 0.4) * -1.0)          
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_lines, validation_lines = train_test_split(lines[1:], test_size=0.2)                                            
                                            
# compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)                                            
#NVIDIA model discussed in classroom                                           
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) #normalization and mean center
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.summary()

model.compile(loss='mse', optimizer='adam') #using Adam optimizer
#fitting model
model.fit_generator(train_generator, steps_per_epoch=int(len(train_lines)/32), 
                    validation_data=validation_generator, validation_steps=int(len(validation_lines)/32), epochs=3, verbose = 1)

model.save('model.h5')

'''        
images = []
measurements = []

#with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #lines.append(line)
        center_image = cv2.imread('data/IMG/' + line[0].split('/')[-1])
        #center_image = cv2.cvtColor(center_image_bgr, cv2.COLOR_BGR2RGB)
        #center_image = ndimage.imread('./data/IMG/' + line[0].split('/')[-1])
        images.append(center_image)
        measurements.append(float(line[3]))
        #flipping images
        images.append(cv2.flip(center_image, 1))
        measurements.append(-float(line[3]))
        
        left_image_bgr = cv2.imread('./data/IMG/' + line[1].split('/')[-1])
        left_image = cv2.cvtColor(left_image_bgr, cv2.COLOR_BGR2RGB)
        #left_image = ndimage.imread('/opt/carnd_p3/data/IMG/' + line[1].split('/')[-1])
        images.append(left_image)
        measurements.append(float(line[3])+0.1)
        #flipping images
        images.append(cv2.flip(left_image, 1))
        measurements.append(-(float(line[3])+0.1))
        
        right_image_bgr = cv2.imread('./data/IMG/' + line[2].split('/')[-1])
        right_image = cv2.cvtColor(right_image_bgr, cv2.COLOR_BGR2RGB)
        #right_image = ndimage.imread('/opt/carnd_p3/data/IMG/' + line[2].split('/')[-1])
        images.append(right_image)
        measurements.append(float(line[3])-0.1)
        #flipping images
        images.append(cv2.flip(right_image, 1))
        measurements.append(-(float(line[3])-0.1))
 '''
