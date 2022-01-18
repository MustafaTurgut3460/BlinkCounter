from itertools import count
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)

# define face mash detector
detector = FaceMeshDetector(maxFaces=1)
# define blink plot in 15-55
plotY = LivePlot(yLimit=[15, 55])
ratioList = []

# define my color
color = (255, 0, 255)

blinkCounter = 0
counter = 0

# mash id list
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

while True:
   success, img = cap.read()
   img = cv2.resize(img, (640, 480))
   
   # detect face mash in frame
   img, faces = detector.findFaceMesh(img)
   
   if faces:
      face = faces[0]
      
      for id in idList:
         cv2.circle(img, face[id], 3, color, cv2.FILLED)
      
      leftUp = face[159]
      leftDown = face[23]
      leftLeft = face[130]
      leftRight = face[243]
      
      # vertical and horizontal lines
      lenghtVer,_ = detector.findDistance(leftUp, leftDown)
      lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
      
      cv2.line(img, leftUp, leftDown, (0, 200, 0), 2)
      cv2.line(img, leftLeft, leftRight, (0, 200, 0), 2)
      
      # eye lenght
      ratio = int((lenghtVer/lenghtHor)*100)
      ratioList.append(ratio)
      
      # for stable results
      if len(ratioList)>3:
         ratioList.pop(0)
         
      ratioAvg = sum(ratioList)/len(ratioList)
      
      #dont count again and again just 1 count for 1 blink, wait a little
      if ratioAvg < 35 and counter == 0:
         blinkCounter += 1
         color = (0, 200, 0)
         counter = 1
         
      if counter != 0:
         counter += 1
         if counter > 10:
            counter = 0
            color = (255, 0, 255)
         
      cv2.putText(img, "Goz Kirpma Sayisi: " + str(blinkCounter), (30, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
      
      # my plot img for blinking
      imgPlot = plotY.update(ratioAvg)
      # stacked 2 images
      stackImage = cvzone.stackImages([img, imgPlot], 2, 1)
   
   else:
      stackImage = cvzone.stackImages([img, img], 2, 1)
      
   cv2.imshow("Video", stackImage)
   cv2.waitKey(1)
