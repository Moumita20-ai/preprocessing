# In[1]
# Remove all text from an image


import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np

#General Approach.....
#Use keras OCR to detect text, define a mask around the text, and inpaint the
#masked regions to remove the text.
#To apply the mask we need to provide the coordinates of the starting and 
#the ending points of the line, and the thickness of the line

#The start point will be the mid-point between the top-left corner and 
#the bottom-left corner of the box. 
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, pipeline):
    # read the image 
    img = keras_ocr.tools.read('0.jpg') 
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([images])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

img_text_removed = inpaint_text('20220228_155359_00000009_00N.JPG', pipeline)

plt.imshow(img_text_removed)

cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))





# In[2]
# Remove selected text from an image 

#import matplotlib.pyplot as plt
import keras_ocr
import cv2
#import math
import numpy as np
from os import listdir



def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


pipeline = keras_ocr.pipeline.Pipeline()


in_path = 'C:/Users/User/Desktop/BEL/Sample/original/'
out_path = 'C:/Users/User/Desktop/BEL/Sample/processed/'

S = listdir(in_path)

for i in range(0,len(S)):
    

    img = keras_ocr.tools.read(in_path + S[i]) 
    image = np.copy(img)
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
# for box in prediction_groups[0]:
    
#         x0, y0 = box[1][0]
#         x1, y1 = box[1][1] 
#         x2, y2 = box[1][2]
#         x3, y3 = box[1][3] 


    for j in range(0,len(prediction_groups[0])):
        QW = prediction_groups[0][j]
        
        if 'north' in QW:
           #for box in QW[1]:  # for 'north'
               
               x0, y0 = QW[1][0] 
               x1, y1 = QW[1][1]
               x2, y2 = QW[1][2]
               x3, y3 = QW[1][3] 
               
               
               x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
               x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
                       
                       #For the line thickness, we will calculate the length of the line between 
                       #the top-left corner and the bottom-left corner.
               thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
                       
                       #Define the line and inpaint
               cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
                       thickness)
               image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                
        elif  'koml' in QW:
           #for box in prediction_groups[0][j]:  # for 'north'
               
               x0, y0 = QW[1][0] 
               x1, y1 = QW[1][1]
               x2, y2 = QW[1][2]
               x3, y3 = QW[1][3]
               
               
               x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
               x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
                       
                       #For the line thickness, we will calculate the length of the line between 
                       #the top-left corner and the bottom-left corner.
               thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
                       
                       #Define the line and inpaint
               cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
                       thickness)
               image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
        
        elif  'patti' in QW:
           #for box in prediction_groups[0][j]:  # for 'north'
               
               x0, y0 = QW[1][0] 
               x1, y1 = QW[1][1]
               x2, y2 = QW[1][2]
               x3, y3 = QW[1][3]
               
               
               x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
               x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
                       
                       #For the line thickness, we will calculate the length of the line between 
                       #the top-left corner and the bottom-left corner.
               thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
                       
                       #Define the line and inpaint
               cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
                       thickness)
               image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)           
                
                
    cv2.putText(image, 'SOUTH WEST', (122,357), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    cv2.putText(image, 'BEL DEMO', (121,	387), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(out_path + S[i], bgr_image)         
    #plt.imshow(image)          
                
            
            

