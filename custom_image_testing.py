import numpy as np
import cv2
import pickle

pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)


# In[5]:
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
 


# In[6]:
imgOriginal = cv2.imread("test2.jpg")
img = np.asarray(imgOriginal)
img = cv2.resize(img, (32, 32))
img = preProcessing(img)
img = img.reshape(1, 32, 32, 1)
classIndex = int(model.predict_classes(img))
predictions = model.predict(img)
probVal= np.amax(predictions)
cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
cv2.imshow("Original Image",imgOriginal)
cv2.waitKey(0)





