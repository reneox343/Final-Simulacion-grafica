
import csv,os
from features import *
directories = ['./Images/Cubism', './Images/Graffiti', './Images/PopArt']
labels = ['Cubism', 'Graffiti', 'PopArt']
e = open('ImagesData.csv', 'w')
writer = csv.writer(e)
writer.writerow(['path','label', 'Features'])
for i in range(len(directories)):
    directory = directories[i]
    label = labels[i]
    for filename in sorted(os.listdir(directory), key=len):
        f = os.path.join(directory, filename)
        f = f.replace("\\","/")
        if os.path.isfile(f):
            image = cv2.imread(f)
            features = featureRedHistogram(image)+featureGreenHistogram(image)+featureBlueHistogram(image)+featureRGBGMedia(image)+featureHSIMedia(image)
            writer.writerow(['{}'.format(f),label,features])