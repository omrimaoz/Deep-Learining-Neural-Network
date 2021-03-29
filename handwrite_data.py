#imports
from PIL import Image
import numpy as np

Handwrite_Dataset = np.zeros(shape =(62,30,40))
letters_digits = []
dic = {}
number =[]
count = 0
length=0
for i in range(123):
    if 47<i<58 or 64<i<91 or 96<i<123:
        letters_digits.append(chr(i))
        dic[chr(i)] = count
        count+=1
    if 0<i<63:
        if(i<10):
            number.append("00"+str(i))
        else:
            number.append("0" + str(i))
count=0
print(number)
for x in letters_digits:
    for num in number[0:55]:
        if 64<ord(x)<91:
            image = Image.open('Hnd/Img/'+x+"./img"+number[count]+"-"+num+".png").convert("L").resize((40,30))
        else:
            image = Image.open('Hnd/Img/'+x+"/img"+number[count]+"-"+num+".png").convert("L").resize((40,30))
        length+=1
    Handwrite_Dataset[count] = np.asarray(image)
    print(count)
    count+=1

Handwrite_Dataset_reshaped = Handwrite_Dataset.reshape(Handwrite_Dataset.shape[0],-1)
np.savetxt("Handwrite_Dataset.txt",Handwrite_Dataset_reshaped)

loaded_arr = np.loadtxt("Handwrite_Dataset.txt")

# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape.reshaping to get original
# matrice with original shape.
load_original_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // Handwrite_Dataset.shape[2], Handwrite_Dataset.shape[2])
print(load_original_arr.shape)
# data = np.asarray(image)

# print(data.shape)
#
# print(image.format)
# print(image.size)
# print(image.mode)

print(length)