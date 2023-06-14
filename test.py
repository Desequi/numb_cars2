import cv2
import easyocr
import numpy as np

# pytesseract.pytesseract.tesseract_cmd = r'<D:\proj>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
img = cv2.imread('__2.jpg')
reader = easyocr.Reader(['en','ru'], gpu = False)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255,0)
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy()

# for idx, contour in enumerate(contours):
#     (x, y, w, h) = cv2.boundingRect(contour)
#     if hierarchy[0][idx][3] == 0:
#         cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

bounds = reader.readtext(img_erode)
print(bounds)
# cv2.imshow("Input", img)
# cv2.imshow("Enlarged", img_erode)
# cv2.imshow("Output", output)
# cv2.waitKey(0)


