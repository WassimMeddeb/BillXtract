import cv2
import pytesseract
import numpy as np
import pandas as pd
from ultralytics import YOLO
import textract
import keras_ocr
import Levenshtein


#Tesseract configuration
custom_words = 'f_dictionnary_v1.txt'
config = f'--psm 6 --user-words {custom_words}'
pipeline = keras_ocr.pipeline.Pipeline()


#cleaning strings
def strip_string(my_string):
    replacements = [('!', ''), ('?', ''),('|',''),("[",""),("]",""),("/",""),('\\',""),('\r',''),('\'',''),('"',''),('',''),("\u200f",""),("\u200e",""),("{",""),("}",""),("%",""),("~",""),(",","."),(':','.'),(";","."),('°',''),('+',''),('(',''),(')','')]
    for char, replacement in replacements:
        if char in my_string:
            my_string = my_string.replace(char, replacement)
    return my_string

def strip_dates(my_string):
    replacements = [('!', ''), ('?', ''),('|',''),("[",""),("]",""),("%",""),("\u200f",""),("\u200e",""),("{",""),("}",""),("&","")]
    for char, replacement in replacements:
        if char in my_string:
            my_string = my_string.replace(char, replacement)
    return my_string

def prepare(my_string,flag):
    if (flag!='paragraph2'):
        tmp=strip_string(my_string)
    else:
        tmp=strip_dates(my_string)
    tmp=tmp.split("\n")
    while("" in tmp):
        tmp.remove("")
    tmp2=tmp
    tmp3=tmp
    if (flag=='tab'):
        tmp2 = [i.replace(' ','') for i in tmp]
        tmp3 = [i.replace('o','0') for i in tmp2]
        #print(tmp3)
        tmp4=' '.join(tmp3)
        return (tmp4)
    return(tmp3)

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
    res = [i for i in letter if i.isdigit()]
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
    replacements = [("a","0"),("p","0"),("q","0"),("d","0"),("o","0"),("O","0"),('e','6')]
    X=string.split(" ")
    for i in X:
        letters=i
        for char, replacement in replacements:
            if char in letters:
                letters = letters.replace(char, replacement)
        corrected.append(letters)
    ss=" ".join(corrected)
    return (ss)



#Correcting words with help of dictionnary
def find_closest_word(word, dictionary_file):
    smallest_distance = float('inf')
    closest_word = ""

    with open(dictionary_file,encoding='utf-8') as f:
        words = [line.strip() for line in f]

    for dict_word in words:
        distance = Levenshtein.distance(word, dict_word)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_word = dict_word

    return closest_word



#function to enhance the contrast of image for better ocr result
def enhance(im):
    img = cv2.imread(im)
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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



#cleaning and extracting data from para 1
def extract_para1(para1):
    parag=prepare(para1[0],'paragraph1')
    LOIs=[parag[0],parag[1]]
    #print(LOIs)
    for LOI in LOIs:
        loi=LOI.split()
        for item in loi:
            try:
                n_facture=int(item)
            except:
                n_facture=0
            if(n_facture > 0 ):
                return (n_facture)
    return(0)
    #print(LOI)


#cleaning and extracting data from para 2
def fix_date(date):
    return find_closest_word(date,'D1/dates.txt')
def extract_para2(para2):
    parag=prepare(para2[0],'paragraph2')
    clean=[]
    for i in parag:
        if len(i)>3:
            clean.append(i)
    #print(clean)
    mois=clean[0].split()
    ref=clean[2].split()
    #print(ref)
    for item in ref:
        try:
            ref_=int(item)
        except:
            ref_=0
        if(ref_ != 0 ):
            break
    for item in mois:
        res = any(chr.isdigit() for chr in item)
        if (res):
            mois_=item
            break
    mois_=fix_date(mois_)
    #print(ref_)
    #print(mois_)
    return([mois_,ref_])



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



#Return the best result from three ocr results
def best_res(string1,string2,string3):
    # Define the three OCR strings
    flag=False
    if (string1==''):
        flag=True
        string1='0123456789.'
    if(string2==''):
        if (flag==False):
            flag=True
            string2='0123456789.'
    if(string3==''):
        if (flag==False):
            flag=True
            string3='0123456789.'
    ocr_strings = [string1,string2,string3]
    #print('st1:',string1,'st2:',string2,'st3:',string3)
    # Calculate the average Levenshtein distance between each OCR string and the other two strings
    avg_distances = []
    for i in range(len(ocr_strings)):
        total_distance = 0
        for j in range(len(ocr_strings)):
            if i != j:
                total_distance += Levenshtein.distance(ocr_strings[i], ocr_strings[j])
        avg_distance = total_distance / 2
        avg_distances.append(avg_distance)

    # Determine the most accurate result based on the average distances
    #if min(avg_distances) < 4:
    index = avg_distances.index(min(avg_distances))
    if len(ocr_strings[index])>0:
        return (ocr_strings[index])
    else:
        return('error')
    #else:
        #return ('not reliable')





def perform_ocr(ocr_im,f,check=False):
    res_keras=[]
    corrected=[]
    res_tes=[]
    tt=cv2.imread(ocr_im)
    if (f == 'ar'):
        res_tess=pytesseract.image_to_string(tt,config=config,lang='ara')
        #res_tes=prepare(res_tes,'tab')
        #print(res_tess)
        res_tess=res_tess.split('\n')
        for x in res_tess:
            if len(x)>1:
                res_tes.append(x)
        #print(res_tes)
        corrected=[]
        for t in res_tes:
            line=''
            tmp=t.split(" ")
            for i in tmp :
                if(containsNumber(i)):
                    line+=i+' '
                else :
                    line+=find_closest_word(i, "D1/arabic.txt")+' '
            corrected.append(line)
        #print(corrected)
        return (corrected)
    elif (f == '*'):
        res_tes=pytesseract.image_to_string(tt,config=config,lang='ara')
        res_tes=prepare(res_tes,'tab')
        if ("تقدير" not in res_tes):
            res_opus=textract.process(ocr_im,method='ocropus')
            res_opus=res_opus.decode("utf-8")
            res_opus=prepare(res_opus,'tab')
            predictions = pipeline.recognize([tt])
            for prediction in predictions[0]:
                res_keras=' '.join(prediction[0])
                res_keras=number_correction(res_keras)
            best=best_res(res_keras,res_tes,res_opus)
            best_s=best.split()
            return (best_s)
        else :
            return (res_tes.split())
    else:
        res_tes=pytesseract.image_to_string(tt,config=config)
        res_tes=prepare(res_tes,'tab')
        res_tes=number_correction(res_tes)
        #print(res_tes)
        res_opus=textract.process(ocr_im,method='ocropus')
        res_opus=res_opus.decode("utf-8")
        res_opus=prepare(res_opus,'tab')
        res_opus=number_correction(res_opus)
        #print(res_opus)
        predictions = pipeline.recognize([tt])
        for prediction in predictions[0]:
            res_keras.append(prepare(prediction[0],'tab'))
        res_keras=' '.join(res_keras)
        if (check==True):
            if("." not in res_keras):
                res_keras=''
        #print(res_keras)
        best=best_res(res_tes,res_opus,res_keras)
        best_s=best.split()
        return (best_s)




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



#performing OCR on the two paragraphs above the table
def ocr_paragraphs(img,results_tab):
    images,classes=extract_bbox(img,results_tab)
    ocr_results=[]
    for C,img in zip(classes,images):
        if (C != 2):
            ocr_results.append([pytesseract.image_to_string(img,config=config),int(C)])
        else :
            cv2.imwrite('D1/frags/sliced.jpg', img)
    for i in ocr_results:
        if (i[1]==1):
            n_facture=extract_para1(i)
        else :
            mois_ref=extract_para2(i)
    #exceptions=save_locally(images,classes)
    return [n_facture,mois_ref]



def save_locally(img,results):
    images,classes=extract_bbox(img,results)
    #storing cropped images locally
    exceptions=[0,0,0,0,0]
    try:
        cv2.imwrite('D1/frags/mont.jpg', images[classes.index(2)])
    except:
        exceptions[2]=1
    try:
        cv2.imwrite('D1/frags/cons.jpg', images[classes.index(0)])
    except:
        exceptions[0]=1
    try:
        cv2.imwrite('D1/frags/indexe.jpg', images[classes.index(4)])
    except:
        exceptions[4]=1
    try:
        cv2.imwrite('D1/frags/month.jpg', images[classes.index(3)])
    except:
        exceptions[3]=1
    try:
        cv2.imwrite('D1/frags/title.jpg', images[classes.index(1)])
        cv2.imwrite('D1/frags/title2.jpg',enhance('title.jpg'))
    except:
        exceptions[1]=1
    return exceptions
    

def performOCR(exceptions,parag_res):
    if(exceptions[4]==0):
        indexe=perform_ocr('D1/frags/indexe.jpg','*')
    else:
        indexe=['error', 'error']
    if(exceptions[0]==0):
        consommation=perform_ocr('D1/frags/cons.jpg','en')
    else:
        consommation=['0' ,'0']
    if(exceptions[1]==0):
        titles=perform_ocr('D1/frags/title2.jpg','ar')
    else:
        titles=['error' ,'error']
    if(exceptions[3]==0):
        month=perform_ocr('D1/frags/month.jpg','en')
    else:
        month=['error','error'] 
    if(exceptions[2]==0):
        montant=perform_ocr('D1/frags/mont.jpg','en',check=True)
    else:
        montant=['0.000','0.000']
    
    s_title =[]
    s_month=[]
    s_indexe =[]
    s_consommation =[]
    s_montant =[]
    i1,i2,i3,i4,i5 = len(titles)-1,len(month)-1,len(indexe)-1,len(consommation)-1,len(montant)-1
    flag=0
    somme=0
    if len(consommation)<=len(titles)/2:
        while((min(i1,i2,i3,i4,i5)>-1)):
            if(i1 % 2==1):
                #print('working1')
                i1-=1
                i2-=1
                somme+=float(montant[i5])
                i5-=1
            else:
                #print('working2')
                s_title.append(titles[i1])
                s_month.append(month[i2])
                s_indexe.append(indexe[i3])
                s_consommation.append(consommation[i4])
                somme+=float(montant[i5])
                s_montant.append(somme)
                somme=0
                i1,i2,i3,i4,i5 = [v - 1 for v in (i1,i2,i3,i4,i5)]
    else:
        while((min(i1,i2,i3,i4)>-1)and(i5> -2)):
            if ("قسط" in titles[i1]):
                flag = 1
                somme += float(montant[i5])
                i1-=1
                i5-=1
                i4-=1
            else :
                if (flag == 0):
                    s_title.append(titles[i1])
                    s_month.append(month[i2])
                    s_indexe.append(indexe[i3])
                    s_consommation.append(consommation[i4])
                    s_montant.append(montant[i5])
                    i5-=1
                else  :
                    s_title.append(titles[i1])
                    s_month.append(month[i2])
                    s_indexe.append(indexe[i3])
                    s_consommation.append(consommation[i4])
                    s_montant.append(somme)
                    somme=0
                    flag=0
                i1,i2,i3,i4 = [v - 1 for v in (i1,i2,i3,i4)]

    df = pd.DataFrame(list(zip(s_title,s_month,s_indexe,s_consommation,s_montant)),columns =['Type', 'Months', 'NV indexe','consommation','montant'])
    df['n_facture']=parag_res[0]
    df['reference']=parag_res[1][1]
    df['date']=parag_res[1][0]
    return df
