#imports
from PIL import Image
import numpy as np

# define variables
Handwrite_Dataset = np.zeros(shape =(62*55,30,40)) # numpy 3'd array
letters_digits = []
dic = {}
number =[]
count = 0
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
i=0
j=0
for x in letters_digits:
    for num in number[0:55]:
        if 64<ord(x)<91:
            image = Image.open('Hnd/Img/'+x+"./img"+number[i]+"-"+num+".png").convert("L").resize((40,30))
        else:
            image = Image.open('Hnd/Img/'+x+"/img"+number[i]+"-"+num+".png").convert("L").resize((40,30))
        Handwrite_Dataset[j] = np.asarray(image)
        j+=1
    i+=1
    print(x)

Handwrite_Dataset_reshaped = Handwrite_Dataset.reshape(Handwrite_Dataset.shape[0],-1)
np.savetxt("Handwrite_Dataset.txt",Handwrite_Dataset_reshaped)

loaded_arr = np.loadtxt("Handwrite_Dataset.txt")

load_original_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // Handwrite_Dataset.shape[2], Handwrite_Dataset.shape[2])
print(load_original_arr.shape)
print(Handwrite_Dataset.shape[2])

