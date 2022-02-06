from PIL import Image
import pytesseract
import numpy as np
import cv2
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
filename = 'images/test_image.jpeg'
filename2 = 'images/test_image_noise.png'
filename3 = 'images/test_image_noise2.png'
def simple_read_data():
    img1 = np.array(Image.open(filename))
    text = pytesseract.image_to_string(img1)
    return text
def simple_read_data_with_noise(filename):
    img = np.array(Image.open(filename))
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    text = pytesseract.image_to_string(img)
    return text
def write_bounding_box_on_text_region():
    image = cv2.imread(filename)
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    print(results)
    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]

        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = float(results["conf"][i])
        if int(conf) > 70:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
    cv2.imshow(image)


if __name__ == '__main__':
    print(simple_read_data())
    print(simple_read_data_with_noise(filename2))
    print(simple_read_data_with_noise(filename3))
    print(write_bounding_box_on_text_region())

