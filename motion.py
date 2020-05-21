import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep

classes = [
    "GO",
    "Slow-down",
    "No gesture",
    "OK"
    ]

num_frames = 0


class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
   
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(4, activation='softmax', name="output")
    

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.out(x)



class VideoCameraMotion(object):
    new_model = Conv3DModel()
    new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop())
    new_model.load_weights('path_to_my_weights2')
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.to_predict = []
        self.classe =''
        self.cls=None
        self.per=None
        
        
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    def normaliz_data(self,np_data):
        scaler = StandardScaler()
        scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
        return scaled_images
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        success, frame = self.video.read()
        frame = cv2.flip(frame,1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.to_predict.append(cv2.resize(gray, (64, 64)))
    
         
        if len(self.to_predict) == 30:
            frame_to_predict = np.array(self.to_predict, dtype=np.float32)
            frame_to_predict = self.normaliz_data(frame_to_predict)
            
            predict = self.new_model.predict(frame_to_predict)
            self.classe = classes[np.argmax(predict)]

            #print('Classe = ',self.classe, 'Precision = ', np.amax(predict)*100,'%')
            self.to_predict = []
        
        cv2.putText(frame, self.classe, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0),1,cv2.LINE_AA)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()