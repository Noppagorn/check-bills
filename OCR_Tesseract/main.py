from PIL import Image
import pytesseract
import numpy as np
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
filename = 'test_image.jpeg'

def simple_read_data():
    img1 = np.array(Image.open(filename))
    text = pytesseract.image_to_string(img1)
    return text

if __name__ == '__main__':
    print(simple_read_data())

