#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import shutil
import csv

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
	#CODE FOR GETTING THE DATA FROM THE 20BNJESTER WEBSITE#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
'''
var x = document.getElementsByClassName("col-md-8")[0].firstChild.nextElementSibling.nextElementSibling.firstChild.nextElementSibling

for(var i=0;i<23;i++){
 var text = x.href;   
   console.log("!wget \"" + text + "\"");
    x = x.nextElementSibling;
    x = x.nextElementSibling; 
    x = x.nextElementSibling;
}'''

#s = ''' paste here'''


'''try:
    while(s.index("VM") != -1):
        index = s.index("VM")
        s = s[:index] + s[index + 7:]
except:
    print(s)
'''

#cat 20bn-jester-v1-* | tar zx
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#








path = "/content/20bn-jester-v1"
dirs = os.listdir(path)


########creation of dictionary from original dataset
targets = pd.read_csv('/content/jester-v1-train.csv', header=None,sep = ";", names=['', 'values'], index_col=0)['values'].to_dict()

#thumb-up
results = []
for k,v in targets.items():
    if v=='Thumb Up':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'


for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

dict1 = {}
for val in results:
  dict1.update( {val : 'Thumb Up'} )



#no-gesture
results = []
for k,v in targets.items():
    if v=='No gesture':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict1.update( {val : 'No gesture'} )



#Swiping Right
for k,v in targets.items():
    if v=='Swiping Right':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict1.update( {val : 'Swiping Right'} )



#Sliding Two Fingers Left
for k,v in targets.items():
    if v=='Sliding Two Fingers Left':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict1.update( {val : 'Sliding Two Fingers Left'} )


#Rolling Hand Forward
for k,v in targets.items():
    if v=='Rolling Hand Forward':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict1.update( {val : 'Rolling Hand Forward'} )


#Zooming Out With Two Fingers
for k,v in targets.items():
    if v=='Zooming Out With Two Fingers':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/training-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict1.update( {val : 'Zooming Out With Two Fingers'} )



##########creation of validation dataset from original data
targets_validation = pd.read_csv('/content/jester-v1-validation.csv', header=None,sep = ";", names=['', 'values'], index_col=0)['values'].to_dict()
results = []

#thumb-up
for k,v in targets_validation.items():
    if v=='Thumb Up':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

dict2 = {}
for val in results:
  dict2.update( {val : 'Thumb Up'} )



#no-gesture
results = []
for k,v in targets_validation.items():
    if v=='No gesture':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict2.update( {val : 'No gesture'} )



#Swiping Right
results = []
for k,v in targets_validation.items():
    if v=='Swiping Right':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict2.update( {val : 'Swiping Right'} )



#Sliding Two Fingers Left
results = []
for k,v in targets_validation.items():
    if v=='Sliding Two Fingers Left':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict2.update( {val : 'Sliding Two Fingers Left'} )


#Rolling Hand Forward
results = []
for k,v in targets_validation.items():
    if v=='Rolling Hand Forward':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict2.update( {val : 'Rolling Hand Forward'} )


#Zooming Out With Two Fingers
results = []
for k,v in targets_validation.items():
    if v=='Zooming Out With Two Fingers':
        for directory in dirs:
                if k == int(directory):
                    results.append(k) 

os.chdir('/content/20bn-jester-v1/')
path = '/content/20bn-jester-v1/'
path1= '/content/validation-data/'

for val in results:
    for directory in dirs:
            if val==int(directory):
                if os.path.isdir(path+directory):
                    shutil.copytree(directory,path1+str(val))

for val in results:
  dict2.update( {val : 'Zooming Out With Two Fingers'} )




print("Number of training data:")
print(len(dict1))

print("Number of validation data:")
print(len(dict2))


#Creating a CSV file for respective DATASETS
with open('/content/training.csv', 'w') as f:
    for key in dict1.keys():
        f.write("%d,%s\n"%(key,dict1[key]))

with open('/content/validation.csv', 'w') as f:
    for key in dict2.keys():
        f.write("%d,%s\n"%(key,dict2[key]))


#Mount google drive#
from google.colab import drive
drive.mount('/content/drive')


#zipping training and validation data directories
!zip -r /content/training-data1.zip  /content/training-data
!zip -r /content/validation-data1.zip  /content/validation-data

#moving dataset and the CSV files to drive
!mv /content/training-data1.zip /content/drive/My\ Drive/Colab\ Notebooks/SLD/
!mv /content/validation-data1.zip /content/drive/My\ Drive/Colab\ Notebooks/SLD/
!mv /content/training.csv /content/drive/My\ Drive/Colab\ Notebooks/SLD
!mv /content/validation.csv /content/drive/My\ Drive/Colab\ Notebooks/SLD





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
			###SWITCH TO GPU MODE IN GOOGLE COLAB###
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import shutil
import csv



from google.colab import drive
drive.mount('/content/drive')

##unzipping the training and validation data from the drive
!unzip /content/drive/My\ Drive/Colab\ Notebooks/SLD/training-data1.zip
!unzip /content/drive/My\ Drive/Colab\ Notebooks/SLD/validation-data1.zip
!cp /content/drive/My\ Drive/Colab\ Notebooks/SLD/training.csv /content/
!cp /content/drive/My\ Drive/Colab\ Notebooks/SLD/validation.csv /content/


targets = pd.read_csv('/content/training.csv', header=None,sep = ",", names=['', 'values'], index_col=0)['values'].to_dict()
targets_validation = pd.read_csv('/content/validation.csv', header=None,sep = ",", names=['', 'values'], index_col=0)['values'].to_dict()





# MODEL CREATION
class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
   
    # LSTM & Flatten
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(6, activation='softmax', name="output")
    
  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    #x = self.pool2(x)
    #x = self.conv3(x)
    #x = self.pool3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.out(x)



model = Conv3DModel()
# choose the loss and optimizer methods
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])




path = "/content/content/training-data/"
path_cv = "/content/content/validation-data"

dirs = os.listdir(path)
dirs_cv = os.listdir(path_cv)
dirs1,dirs_cv1 = os.listdir(path),os.listdir(path_cv)
for i in range(0, len(dirs)): 
    dirs1[i]=int(dirs[i])
for j in range(0, len(dirs_cv)): 
    dirs_cv1[j]=int(dirs_cv[j])


n = [0] * (max(list(targets.keys()))+10) #Training List maintained to ensure no directory is considered more than once
m = [0] * (max(list(targets_validation.keys()))+10)
print(len(dirs))
print(len(dirs_cv))
print(dirs)
print(dirs_cv)
print(len(n))
print(len(m))



###IMAGE processing code####
#return gray image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
# Resize frames
def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame

hm_frames = 30 
def get_unify_frames(path):
    offset = 0
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames 
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        #diff = (frames_count - hm_frames)
        #offset = diff-1 
        frames = frames[0:hm_frames]
    return frames  



targets_name = ["Thumb Up","No gesture","Swiping Right","Sliding Two Fingers Left","Rolling Hand Forward","Zooming Out With Two Fingers"]



!rm -r /content/train
!rm -r /content/validation1


for iteration in range(10):
  #------------------------------------------------------PROCESSING AND PREPARING THE TRAINING DATA-----------------------------------------------------#

  counter_train = 0
  results_train = []
  batch_dict = {}
  results_train = []
  path = "/content/content/training-data/"
  dirs = os.listdir(path)
  for k,v in targets.items():
    if counter_train != 2500  and n[k]==0:
      if v=='No gesture' or 'Swiping Right' or 'Sliding Two Fingers Left' or 'Thumb Up' or 'Rolling Hand Forward' or 'Zooming Out With Two Fingers':
        for directory in dirs:
          if k == int(directory):
            batch_dict.update({k:v})
            n[k]=1
            results_train.append(k)
            counter_train += 1  
  # print(len(results_train))
  # print(batch_dict)
  !rm -r /content/train

  import shutil
  os.chdir('/content/content/training-data/')
  path = '/content/content/training-data/'
  path1= '/content/train/'
  for val in results_train:
      for directory in dirs:
              if val==int(directory):
                  if os.path.isdir(path+directory):
                      shutil.copytree(directory,path1+str(val))

  # Adjust training data
  dirs = os.listdir(path1)
  counter_training = 0 # number of training
  training_targets = [] # training targets 
  new_frames = [] # training data after resize & unify
  for directory in dirs:
      new_frame = [] # one training
      # Frames in each folder
      frames = get_unify_frames(path1+directory)
      if len(frames) == hm_frames: # just to be sure
          for frame in frames:
              frame = resize_frame(path1+directory+'/'+frame)
              new_frame.append(rgb2gray(frame))
              if len(new_frame) == 15: # partition each training on two trainings.
                  new_frames.append(new_frame) # append each partition to training data
                  training_targets.append(targets_name.index(batch_dict[int(directory)]))
                  counter_training +=1
                  new_frame = []


  #show data
  fig = plt.figure()
  for i in range(2,4):
      for num,frame in enumerate(new_frames[i][0:18]):
          y = fig.add_subplot(4,5,num+1)
          y.imshow(frame, cmap='gray')
      fig = plt.figure()
  plt.show()


  # convert training data to np float32
  training_data = np.array(new_frames[0:counter_training], dtype=np.float32)

  #print shape
  print(training_data.shape)



  # Normalisation: training
  print('old mean', training_data.mean())
  scaler = StandardScaler()
  scaled_images  = scaler.fit_transform(training_data.reshape(-1, 15*64*64))
  print('new mean', scaled_images.mean())
  scaled_images  = scaled_images.reshape(-1, 15, 64, 64, 1)
  print(scaled_images.shape)




  #------------------------------------------------------------------------------------------------------------------------------------------------------#




  #------------------------------------------------------PROCESSING AND PREPARING THE VALIDATION DATA-----------------------------------------------------#


  counter_validate = 0
  results_validation = []
  batch_dict_validation = {}
  results_validation = []
  path_cv = "/content/content/validation-data/"
  dirs_cv = os.listdir(path_cv)
  for k,v in targets_validation.items():
    if counter_validate != 310   and m[k]==0:
      if v=='No gesture' or 'Swiping Right' or 'Sliding Two Fingers Left' or 'Thumb Up' or 'Rolling Hand Forward' or 'Zooming Out With Two Fingers':
        for directory in dirs_cv:
          if k == int(directory):
            batch_dict_validation.update({k:v})
            m[k]=1
            results_validation.append(k)
            counter_validate += 1  
  print(len(results_validation))
  !rm -r /content/validation1
  import shutil
  os.chdir('/content/content/validation-data/')
  path = '/content/content/validation-data/'
  path1_cv= '/content/validation1/'
  for val in results_validation:
      for directory in dirs_cv:
              if val==int(directory):
                  if os.path.isdir(path+directory):
                      shutil.copytree(directory,path1_cv+str(val))


  # Adjust validation data
  dirs_cv = os.listdir(path1_cv)
  counter_validation = 0
  cv_targets = []
  new_frames_cv = []
  for directory in dirs_cv:
      new_frame = []
      # Frames in each folder
      frames = get_unify_frames(path1_cv+directory)
      if len(frames)==hm_frames:
          for frame in frames:
              frame = resize_frame(path1_cv+directory+'/'+frame)
              new_frame.append(rgb2gray(frame))
              if len(new_frame) == 15:
                  new_frames_cv.append(new_frame)
                  cv_targets.append(targets_name.index(batch_dict_validation[int(directory)]))
                  counter_validation +=1
                  new_frame = []


  # convert validation data to np float32
  cv_data = np.array(new_frames_cv[0:counter_validation], dtype=np.float32)




  #print shape
  print(cv_data.shape)



  # Normalisation: validation
  print('old mean', cv_data.mean())
  scaler = StandardScaler()
  scaled_images_cv  = scaler.fit_transform(cv_data.reshape(-1, 15*64*64))
  print('new mean',scaled_images_cv.mean())
  scaled_images_cv  = scaled_images_cv.reshape(-1, 15, 64, 64, 1)
  print(scaled_images_cv.shape)






  #------------------------------------------------------------------------------------------------------------------------------------------------------#
  #TRANSFER DATA TO NUMPY ARRAYS
  x_train = np.array(scaled_images)
  y_train = np.array(training_targets)
  x_val = np.array(scaled_images_cv)
  y_val = np.array(cv_targets)
  print(x_train.shape)
  print(y_train.shape)
  print(x_val.shape)
  print(y_val.shape)



  # Run the training 
  history = model.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=32,
                      epochs=2,verbose=1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


  if iteration==9:
    print('Finished')






#SAVING THE WEIGHTS#
model.save_weights('/content/WEIGHTS_6GESTURES', save_format='tf')
