import csv
import cv2
import numpy as np
import os
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augmented_images, augmented_angles = [],[]
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    images.append(image)
                    if i == 0:
                        angles.append(angle)
                    elif i == 1:
                        angles.append(angle+0.2)
                    else:
                        angles.append(angle-0.2)

            ##Get the augmented data
            #augmented_images, augmented_angles = [],[]
            for image,angle in zip (images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

##Import the csv file into the program
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
lines = lines[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

##Convert the path of every pictures for AWS server
'''
images = []
measurements = []
for line in lines[1:]:
	for i in range(3):	
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/'+filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		#For center
		if i == 0:
			measurements.append(measurement)
		#For left
		elif i == 1:
			measurements.append(measurement+0.2)
		#For right
		else:
			measurements.append(measurement-0.2)
'''

##Get the augmented data
'''
augmented_images, augmented_measurements = [],[]
for image,measurement in zip (images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)
'''

##Input the data to numpy array
'''
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#X_train = np.array(images)
#y_train = np.array(measurements)
'''

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


##Create a LeNet model
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
##model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=5)
'''
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
'''
model.save('model.h5')
