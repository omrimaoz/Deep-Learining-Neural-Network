#imports:
import numpy as np
import zlib


def decompress_zlibFile(filename):
    # load and decompress data from 'compressed_Dataset.zlib file
    compressedText = open(filename + ".zlib", "rb").read()

    decompressedText = zlib.decompress(compressedText)
    decompressed_arr = np.frombuffer(decompressedText, np.uint8)

    #stage screen massage:
    print("Successfully decompressed dataset\n")

    # reshape the data into 3d numpy array - shape=(104000,28,28)
    # 104000 images constructed the dataset
    # 28x28 is the image ratio
    dataset = decompressed_arr.reshape(104000, 28, 28)
    return dataset