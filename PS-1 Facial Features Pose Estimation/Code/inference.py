import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

#Load the fine tuned model
model = YOLO('best.pt')

image_path = '..\images\img1920_png.rf.43f14eac933f924af250122f6b74e456.jpg'

#Read the image and convert to RGB
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

#Perform Inference with the fine tuned model
results = model(image_path)


# Iterate over all results and plot the detected bounding box and Landmarks
for result in results:

    for landmarks, box in zip(result.keypoints.xy, result.boxes.data.tolist()): 

        x1,y1,x2,y2,score,class_id = box
                
        x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)

        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)

        for landmark in landmarks:
            if landmark[0]!=0 and landmark[1]!=0:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(image_rgb, (x, y), 3, (255, 0, 0), cv2.FILLED) 
       
#Display and save the image
output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(r'output2.png', output)
cv2.imshow('output', output)
cv2.waitKey(0)


