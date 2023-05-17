import cv2
#import csv
import numpy as np
import pandas as pd
from ultralytics import YOLO
#from PIL import Image
import matplotlib.pyplot as plt
import keras_ocr
import Levenshtein
from skimage.filters import threshold_local
import easyocr



text_reader = easyocr.Reader(['en']) #Initialzing the ocr
config = f'--psm 6'


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255
#function to enhance the contrast of image for better ocr result
def enhance(im):
    img1 = cv2.imread(im)
    img = bw_scanner(img1)
    # converting to LAB color space
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #result = np.hstack((img, enhanced_img))
    #cv2.imshow('Result', result)
    #cv2.waitKey(0)
    return enhanced_img


# extracting bounding boxes cordinates and their labels
def boxes_labels(results,x,y,h,w,c):
    bbox=results[0].boxes.boxes
    bbox=bbox.cpu().numpy() 
    boxes=results[0].boxes.xyxy
    for i,box in enumerate(boxes):
        box=box.cpu().numpy()
        print("fragment ",i," :",box)
        x.append(int(box[0]))
        y.append(int(box[1]))
        h.append(int(box[3]-y[i]))
        w.append(int(box[2]-x[i]))
        c.append(bbox[i][5])
    return 0


#storing ROIs in a liste
def crop_image(img,x,y,h,w):
    cropped=[]
    for (X,Y,H,W) in zip(x,y,h,w):
        cropped.append(img[Y:Y+H, X:X+W])
    print("cropped images: ",len(cropped))
    return (cropped)


#to avoid the presence of multi instance from classes
def singilarity(cropped,c,classes,images):
   # images=[]
    for (img,C) in zip(cropped,c):
        try:
            i=classes.index(C)
        except:
            i=-1
        if i<0 :
            classes.append(C)
            images.append(img)
        else :
            if len(img)>len(images[i]) :
                images[i]=img
    print('number of images:',len(images),'number of classes',len(classes))



#cleaning strings
def strip_string(my_string):
    replacements = [('!', ''), ('?', ''),('|',''),("[",""),("]",""),('\\',""),('\r',''),('\'',''),('.',''),('',''),("\u200f",""),("\u200e",""),("{",""),("}",""),("%",""),("~",""),(",","."),(':','.'),(";","."),('°',''),('+',''),('(',''),(')','')]
    for char, replacement in replacements:
        if char in my_string:
            my_string = my_string.replace(char, replacement)
    return my_string

def prepare(my_string,flag):
    tmp=strip_string(my_string)
    tmp=tmp.split("\n")
    while("" in tmp):
        tmp.remove("")
    return(tmp)

def remove_substring_except_last(string, substring):
    # Replace all occurrences of the substring except the last one with an empty string
    new_string = string.replace(substring, "", string.count(substring)-1)
    # Split the string at the last occurrence of the substring
    parts = new_string.rsplit(substring, 1)
    # Join the parts back together with the substring
    return substring.join(parts)

def reglage_montant(money):
    for i in range(len(money)):
        money[i]=remove_substring_except_last(money[i],'.')
    return (money)

def get_numbers(test_string):
    letter = [x for x in test_string]
    res = [i for i in letter if (i.isdigit() or i=='.')]
    result=''.join(res)
    if (len(result)>0):
        return result
    else:
        return (test_string)
def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False
def number_correction(string):
    corrected=[]
    replacements = [("Q","0"),("p","0"),("q","0"),("d","0"),("o","0"),("O","0"),('I','1'),('z','2'),('Z','2'),('a','3'),(" ",""),("B","8")]
    X=string.split(" ")
    for i in X:
        letters=i
        for char, replacement in replacements:
            if char in letters:
                letters = letters.replace(char, replacement)
        corrected.append(letters)
    ss=" ".join(corrected)
    return (ss)





def divide_image(im):
    image = cv2.imread(im)

    height, width = image.shape[:2]
    midpoint = width // 2

    left_image = image[:, :midpoint].copy()
    right_image = image[:, midpoint:].copy()
    return([left_image,right_image])


def easy_oc(image):
    results = text_reader.readtext(image )
    textt=[]
    for (bbox, text, prob) in results:
        textt.append(text)
    easy_res=''.join(textt)
    return easy_res


import re
def is_valid_number(string):
    pattern = r'^[0-9,.-]+$'
    return re.match(pattern, string) is not None and len(string) > 2
def is_valid_date(string):
    pattern = r'^(0[1-9]|1[0-2])/(\d{4})$'
    return re.match(pattern, string) is not None

#performing OCR on the ROI using the easyOCR engine
def simple_ocr(im,seek_date=False):
    imss=cv2.imread(im)
    results = text_reader.readtext(imss )
    textt=[]
    numbers=[]
    seeked_date=''
    for (bbox, text, prob) in results:
        textt.append(text)
    #print(textt)
    for word in textt:
        if is_valid_number(word):
            numbers.append(word)
    if seek_date==False:
            return numbers
    else:
        for word in textt:
            if is_valid_date(word):
                seeked_date=word
        return(numbers,seeked_date)



def extract_bbox(img,results_tab):
    #Extracting bounding boxes and ROI from the table model
    x=[]
    y=[]
    h=[]
    w=[]
    c=[]
    cropped=[]
    boxes_labels(results_tab,x,y,h,w,c)
    cropped=crop_image(img,x,y,h,w)
    classes=[]
    images=[]
    singilarity(cropped,c,classes,images)
    return(images,classes)



def save_locally(imagess,classess):
    #storing cropped images locally
    exceptions=[0,0,0,0,0,0,0]
    try:
        cv2.imwrite('D4/frags/bon.jpg', imagess[classess.index(0)])
        cv2.imwrite('D4/frags/bonus.jpg', enhance('D4/frags/bon.jpg'))
    except:
        exceptions[0]=1
    try:    
        cv2.imwrite('D4/frags/con.jpg', imagess[classess.index(1)])
        cv2.imwrite('D4/frags/consommation.jpg', enhance('D4/frags/con.jpg'))
    except:
        exceptions[1]=1
    try:
        cv2.imwrite('D4/frags/n_d.jpg', imagess[classess.index(2)])
        cv2.imwrite('D4/frags/num_date.jpg', enhance('D4/frags/n_d.jpg'))
    
    except:
        exceptions[2]=1
    try:
        cv2.imwrite('D4/frags/pen.jpg', imagess[classess.index(3)])
        cv2.imwrite('D4/frags/penalty.jpg', enhance('D4/frags/pen.jpg'))
    except:
        exceptions[3]=1

    try:
        cv2.imwrite('D4/frags/per.jpg', imagess[classess.index(4)])
        cv2.imwrite('D4/frags/periodique.jpg', enhance('D4/frags/per.jpg'))
    except:
        exceptions[4]=1
    try:
        cv2.imwrite('D4/frags/ref.jpg', imagess[classess.index(5)])
        cv2.imwrite('D4/frags/reference.jpg', enhance('D4/frags/ref.jpg'))
    except:
        exceptions[5]=1
    try:
        cv2.imwrite('D4/frags/tri.jpg', imagess[classess.index(6)])
        cv2.imwrite('D4/frags/triphase.jpg', enhance('D4/frags/tri.jpg'))
    except:
        exceptions[6]=1
    #slicing images containing two columns
    old_tri,new_tri=divide_image('D4/frags/triphase.jpg')
    old_per,new_per=divide_image('D4/frags/periodique.jpg')
    cv2.imwrite('D4/frags/old_triphase.jpg',old_tri)
    cv2.imwrite('D4/frags/new_triphase.jpg',new_tri)
    cv2.imwrite('D4/frags/old_periodique.jpg',old_per)
    cv2.imwrite('D4/frags/new_periodique.jpg',new_per)
    return exceptions



def perform_ocr():
    #performing OCR on each image fragment
    reference=simple_ocr('D4/frags/reference.jpg')
    print('reference: ',reference)
    num_fact,date_fact=simple_ocr('D4/frags/num_date.jpg',seek_date=True)
    print('numero facture: ',num_fact)
    print('date facture',date_fact)
    new_periodique=simple_ocr('D4/frags/new_periodique.jpg')
    old_periodique=simple_ocr('D4/frags/old_periodique.jpg')
    new_triphase=simple_ocr('D4/frags/new_triphase.jpg')
    old_triphase=simple_ocr('D4/frags/old_triphase.jpg')
    consommation=simple_ocr('D4/frags/con.jpg')
    print('consommation: ',consommation)
    bonification=simple_ocr('D4/frags/bonus.jpg')
    penalty=simple_ocr('D4/frags/penalty.jpg')
    if(len(penalty)==0):
        penalty=[0]
    print('penalite: ',penalty)
    if(len(bonification)==0):
        bonification=[0]
    print('Bonification: ',bonification)
    x=len(new_triphase)
    y=len(new_periodique)
    if x<2:
        for i in range(3-x):
            new_triphase.append(0)
            old_triphase.append(0)
    if y<2:
        for i in range(4-y):
            new_periodique.append(0)
            old_periodique.append(0)

    print('index triphasé ancient: ',old_triphase)
    print('index triphasé nouveau: ',new_triphase)
    print('index periodique ancient: ',old_periodique)
    print('index periodique nouveau: ',new_periodique)

    # Create the nested dictionary using lists for keys and values
    nested_dict = {
        'reference':reference[0],
        'numero_fact': num_fact[0],
        'date_facture':date_fact,
        'consommation':consommation[0],
        'bonification':bonification[0],
        'penalite':penalty[0],
        'triphase':{
            'phase_1':[old_triphase[0],new_triphase[0]],
            'phase_2':[old_triphase[1],new_triphase[1]],
            'phase_3':[old_triphase[2],new_triphase[2]]
        },
        'periodique': {
            'jour': [old_periodique[0],new_periodique[0]],
            'pointe':[old_periodique[1],new_periodique[1]],
            'soir':[old_periodique[2],new_periodique[2]],
            'nuit':[old_periodique[3],new_periodique[3]]
        }
        }   
    #return(reference,num_fact,date_fact,new_periodique,old_periodique,new_triphase,old_triphase,consommation,bonification,penalty)
    return nested_dict