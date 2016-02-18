# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:13:49 2016

@author: Marko Ivetic
"""


# IMPORT


#import potrebnih biblioteka za K-means algoritam

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
iris = datasets.load_iris() #Iris dataset koji će se koristiti kao primer https://en.wikipedia.org/wiki/Iris_flower_data_set

#import potrebnih biblioteka
import cv2
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

#gui
from Tkinter import*
from tkFileDialog import askopenfilename
 

import ttk
import PIL
from PIL import Image, ImageTk, ImageDraw
#gui

#Funkcionalnost implementirana u V1
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def erodeStronger(image):
    kernel = np.ones((6,6)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

#Funkcionalnost implementirana u V2
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def adaptivniMean(image_ada,prviParametar,drugiParametar):
    image_ada_bin = cv2.adaptiveThreshold(image_ada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,prviParametar, drugiParametar)
    return image_ada_bin

def adaptivniGaus(image_ada,prviParametar,drugiParametar):
    image_ada_bin = cv2.adaptiveThreshold(image_ada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,prviParametar, drugiParametar)
    return image_ada_bin
# adaptivni threshold gde se prag racuna = tezinska suma okolnih piksela, gde su tezine iz gausove raspodele


def create_ann1():
    
    ann = Sequential()
    # Postavljanje slojeva neurona mreže 'ann'
    ann.add(Dense(input_dim=784, output_dim=128,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    ann.add(Dense(input_dim=128, output_dim=8,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    return ann
    
def train_ann1(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=500, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
      
    return ann

#Naredna funkcija(select roi) vraca 2 povratnu vrednost koja je dictionary ( regions_dictY) koji sadrzi liste regiona po redovima
#primer regions_dictY[3] daje listu regiona u 4. redu. onda recimo prvu konturu u 4. redu bi dobio kao regions_dictY[3][0]
def select_roi(image_orig, image_bin,lineNumber):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    height,width=image_orig.shape[0:2]
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_dict = {}
    regions_dictY = {}
    listaY=list()
    print len(contours)
    yOld=0
    i=0
    for contour in contours: 
        
        x,y,w,h = cv2.boundingRect(contour)
        if w>8 and h<100 and w*h<1500 and h>w and h<w*2.8 and x<width/2:
            region = image_bin[y:y+h+1,x:x+w+1];
            # Proširiti regions_dict elemente sa vrednostima boundingRect-a ili samim konturama
            regions_dict[x] = [resize_region(region), (x,y,w,h)]
            
            
            if abs(y-yOld)<30:
                
                listaY.append([resize_region(region), (x,y,w,h)])
                #print x,y,yOld
            else:
                #print "duzina liste" + str(len(listaY))
                
                regions_dictY[i] = copy.deepcopy(listaY)
                listaY=list()
                listaY.append([resize_region(region), (x,y,w,h)])
                
                i=i+1
                yOld=y
            if i==4:
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            elif i==6:
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,0,0),2)
            elif i==7:
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,100,100),2)
    #
    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    sorted_regionsY=np.array(regions_dictY[lineNumber])
    #
    #print "Broj regiona" + str(len(regions_dictY))
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    #print "y duzina" + str(len(regions_dictY))
    
    return image_orig, regions_dictY[lineNumber], region_distances
    
#pomocna funkcija koja vraca x komponentu regiona
def getXComponentOfRegion(region):
    return region[1][0]
    
#list of regions = lista regiona u dictionary-u
# sortira najjednostavnije sve regione u toj listi ( red na slici) po x komponenti u rastucem poretku
def sortRegionListByXComponent(listOfRegions):
    for i in range(0,len(listOfRegions)-1):
        for j in range(i+1,len(listOfRegions)):
            if (getXComponentOfRegion(listOfRegions[j]) < getXComponentOfRegion(listOfRegions[i]) ):
                    temp=listOfRegions[i]
                    listOfRegions[i]=listOfRegions[j]
                    listOfRegions[j]=temp
                    
def select_roi1(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    height,width=image_orig.shape[0:2]
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_dict = {}
    print len(contours)
    for contour in contours: 
        
        x,y,w,h = cv2.boundingRect(contour)
        if w>20 and h>20:
            region = image_bin[y:y+h+1,x:x+w+1];
            # Proširiti regions_dict elemente sa vrednostima boundingRect-a ili samim konturama
            regions_dict[x] = [resize_region(region), (x,y,w,h)]
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, sorted_regions[:, 0], region_distances
    
    
#njihova metoda za prikaze rezultata rada neuronske mreze
def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    #w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        #if (k_means.labels_[idx] == w_space_group):
            #result += ' '
        result += alphabet[winner(output)]
    return result
    
def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=500, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
    return ann

def openFile(ann,alphabet):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    location = askopenfilename() # show an "Open" dialog box and return the path to the selected file
  #  radiSve(location)
    dugmeSacuvaj = Button(root, text="Upisi podatke", width=20, height=5, command=lambda:prepoznavanje(ann,alphabet))
    dugmeSacuvaj.place(x=640,y=620)
    
    return location
   
   
def set_Pic_path(text):
    img = Image.open(text) # da taj image uzme tu sliku sa te pozicije
    #img = img.resize((800, 600), PIL.Image.ANTIALIAS) # resizuj tu sliku
    
    img.save('resized.jpg') # sejvuj sliku, da svaki put bude isto ime
  
    photo = ImageTk.PhotoImage(Image.open('resized.jpg')) # da je prikazemo u gui-ju
    labelSlika = Label(root, image=photo) # labela da bi ucitali sliku
    labelSlika.image = photo
    labelSlika.place(x=20, y=20)
       
   
    return
    
def obucavanje():
    
    #display_image(regioniSlika)
    ucitanaSlika1 = load_image("ob.jpg")
    #display_image(ucitanaSlika1)  NE
    sivaSlika1 = image_gray(ucitanaSlika1)
    #display_image(sivaSlika1) NE
    crnoBela1 = image_bin(sivaSlika1)
    #display_image(crnoBela1) NE
    adaptivnaSlika1 = adaptivniGaus(sivaSlika1,45,19)
    #display_image(adaptivnaSlika1) NE
    erozija1 = erode(adaptivnaSlika1)
    dilatacija21=dilate(erozija1)
    #display_image(dilatacija21) NE
    
    regioniSlika1,regioni1,distance1=select_roi1(ucitanaSlika1.copy(),invert(crnoBela1))
    #display_image(regioniSlika1)
    
    alphabet = ['1', '2', '3', '4', '5', '6', '8', '9', '0', '7'] 
    
    
    inputs = prepare_for_ann(regioni1)
    #inputs2 = prepare_for_ann(regioni)
    outputs = convert_output(alphabet)
    #kreiranje neuronske mreze
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)
        
    return ann, alphabet
    

def prepoznavanje(ann,alphabet):
    
    print "USAO U PREPOZNAVANJE"
    
    ucitanaSlika = load_image('resized.jpg')
    sivaSlika = image_gray(ucitanaSlika)
    adaptivnaSlika = adaptivniGaus(sivaSlika,45,19)
    erozija = erode(adaptivnaSlika)
    dilatacija2=dilate(erozija)
    
    #sortiraj liste regiona po x
    regioniSlika,regioni4,distance=select_roi(ucitanaSlika.copy(),invert(dilatacija2),4)

    #for i in range(0,len(regioni4)):
    
    sortRegionListByXComponent(regioni4)
    
    
    regioniSlika2,regioni7,distance7=select_roi(ucitanaSlika.copy(),invert(dilatacija2),7)

    #for i in range(0,len(regioni7)):
    
    sortRegionListByXComponent(regioni7)
    
    
    regioniSlika2,regioni6,distance7=select_roi(ucitanaSlika.copy(),invert(dilatacija2),6)

    #for i in range(0,len(regioni6)):
        
    sortRegionListByXComponent(regioni6)
    
    
    ####
    
    

    lista4=[]

    for x in range(len(regioni4)):
        lista4.append(regioni4[x][0])
    

    lista6=[]

    for x in range(len(regioni6)):
        lista6.append(regioni6[x][0])
        
        
    lista7=[]
        
    for x in range(len(regioni7)):
        lista7.append(regioni7[x][0])
        
        
   
            
    inputs_test4 = prepare_for_ann(lista4)
    results_test4 = ann.predict(np.array(inputs_test4, np.float32))
            
    inputs_test7 = prepare_for_ann(lista7)
    results_test7 = ann.predict(np.array(inputs_test7, np.float32))
            
    inputs_test6 = prepare_for_ann(lista6)
    results_test6 = ann.predict(np.array(inputs_test6, np.float32))
            
    string = ''
    jmbg = ''
            
    jmbg = display_result(results_test4, alphabet, [])
            
    print 'jmbg je ' + display_result(results_test4, alphabet, [])
            
    karta=display_result(results_test7, alphabet, [])
            
    print 'broj karte je ' + karta[0:5]
            
    zona=display_result(results_test6, alphabet,[])
    print 'zona je  ' +zona[0]
            
    string = jmbg + "     " + karta[0:5] + "        " + zona[0]
            
            
            

    f = open('Dokument','a')
    f.write(string + '\n') 
    f.close() 


#ucitanaSlika = load_image("ale2.jpg")
#display_image(ucitanaSlika)

#sivaSlika = image_gray(ucitanaSlika)
#display_image(sivaSlika)

#adaptivnaSlika = adaptivniGaus(sivaSlika,45,19)
#display_image(adaptivnaSlika)

#erozija = erode(adaptivnaSlika)
#dilatacija2=dilate(erozija)
#display_image(dilatacija2)

#sortiraj liste regiona po x
#regioniSlika,regioni4,distance=select_roi(ucitanaSlika.copy(),invert(dilatacija2),4)

#for i in range(0,len(regioni4)):
    
#sortRegionListByXComponent(regioni4)
    
    
#regioniSlika2,regioni7,distance7=select_roi(ucitanaSlika.copy(),invert(dilatacija2),7)

#for i in range(0,len(regioni7)):
    
#sortRegionListByXComponent(regioni7)
    
    
#regioniSlika2,regioni6,distance7=select_roi(ucitanaSlika.copy(),invert(dilatacija2),6)

#for i in range(0,len(regioni6)):
    
#sortRegionListByXComponent(regioni6)


####





location = ''


ann, alphabet = obucavanje();
root = Tk()
root.resizable(0,0)
root.title("Aplikacija za preuzimanje podataka sa vozne karte JGSPNS")
root.geometry('870x700+200+200')

dugmeUcitaj = Button(root, text="Izberite sliku", width=20, height=5, command=lambda:set_Pic_path(openFile(ann, alphabet)))
dugmeUcitaj.place(x=30,y=620)   


root.mainloop()
    