import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep



img_counter = 0

image_x, image_y = 64,64



classes = [
    "move left",
    "move right",
    "No gesture",
    "Thumbs Up"
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
    
    flag=None


    from tensorflow.keras.models import load_model
    classifier = load_model('model.h5')
    img_text = ''

    new_model = Conv3DModel()
    new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop())
    new_model.load_weights('path_to_my_weights2')
   
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        self.to_predict = []
        self.classe =''
        self.cls=None
        self.per=None

    def predictor(self):
       import numpy as np
       from tensorflow.keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       self.result = self.classifier.predict(test_image)
       
       if self.result[0][0] == 1:
              return 'A'
       elif self.result[0][1] == 1:
              return 'B'
       elif self.result[0][2] == 1:
              return 'C'
       elif self.result[0][3] == 1:
              return 'D'
       elif self.result[0][4] == 1:
              return 'E'
       elif self.result[0][5] == 1:
              return 'F'
       elif self.result[0][6] == 1:
              return 'G'
       elif self.result[0][7] == 1:
              return 'H'
       elif self.result[0][8] == 1:
              return 'I'
       elif self.result[0][9] == 1:
              return 'J'
       elif self.result[0][10] == 1:
              return 'K'
       elif self.result[0][11] == 1:
              return 'L'
       elif self.result[0][12] == 1:
              return 'M'
       elif self.result[0][13] == 1:
              return 'N'
       elif self.result[0][14] == 1:
              return 'O'
       elif self.result[0][15] == 1:
              return 'P'
       elif self.result[0][16] == 1:
              return 'Q'
       elif self.result[0][17] == 1:
              return 'R'
       elif self.result[0][18] == 1:
              return 'S'
       elif self.result[0][19] == 1:
              return 'T'
       elif self.result[0][20] == 1:
              return 'U'
       elif self.result[0][21] == 1:
              return 'V'
       elif self.result[0][22] == 1:
              return 'W'
       elif self.result[0][23] == 1:
              return 'X'
       elif self.result[0][24] == 1:
              return 'Y'
       elif self.result[0][25] == 1:
              return 'Z'
       else:
              return 'No gesture'
       
       
   
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
      
        if(self.flag == True):
            frame = cv2.flip(frame,1)
        
            img = cv2.rectangle(frame, (800,100),(1200,500), (0,255,0), thickness=1, lineType=8, shift=0)
        
          
            lower_blue = np.array([0, 0, 183])
            upper_blue = np.array([179, 255, 255])
            imcrop = img[102:498, 802:1198]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
           
           
       
            cv2.putText(frame, self.img_text, (30, 100), cv2.FONT_ITALIC, 2.2, (224, 37, 20),7,cv2.LINE_AA)
            img_name = "1.png"
            save_img = cv2.resize(mask, (image_x, image_y))
            cv2.imwrite(img_name, save_img)
            #print("{} written!".format(img_name))
            self.img_text = self.predictor()
            
           

           
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        else:
            frame = cv2.flip(frame,1)
           
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.to_predict.append(cv2.resize(gray, (64, 64)))
       
             
            if len(self.to_predict) == 30:
                frame_to_predict = np.array(self.to_predict, dtype=np.float32)
                frame_to_predict = self.normaliz_data(frame_to_predict)
               
                predict = self.new_model.predict(frame_to_predict)
                self.classe = classes[np.argmax(predict)]

                print('Classe = ',self.classe, 'Precision = ', np.amax(predict)*100,'%')
                self.to_predict = []
           
            cv2.putText(frame, self.classe, (30, 100), cv2.FONT_ITALIC, 1.9, (224, 37, 20),7,cv2.LINE_AA)
           
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
