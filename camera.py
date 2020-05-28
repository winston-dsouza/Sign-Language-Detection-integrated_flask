import cv2
import numpy as np


img_counter = 0

image_x, image_y = 64,64





class VideoCamera(object):
    from tensorflow.keras.models import load_model
    classifier = load_model('model.h5')
    img_text = ''
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    
    def __del__(self):
        self.video.release()
    
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
    
    def get_frame(self):
        
        success, frame = self.video.read()
        frame = cv2.flip(frame,1)
        
        img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
        lower_blue = np.array([0, 0, 161])
        upper_blue = np.array([179, 255, 255])
        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        
        print(self.img_text)
        cv2.putText(frame, self.img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0),1,cv2.LINE_AA)
        img_name = "1.png"
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))
        self.img_text = self.predictor()
        

        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
