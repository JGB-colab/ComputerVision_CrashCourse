import cv2
import imutils
img = r'day2\big_hero.jpg'
image  = cv2.imread(img)
(h,w,d) = image.shape
class basic_opencv():
    def show():
        print("width={},height = {},depth = {}".format(w,h,d))
        #cv2.imshow("image",image)
        #cv2.waitKey(0)
    def show_img(rotulo,image):
        cv2.imshow(rotulo,image)
        cv2.waitKey(0)

    def segmentation_pixel():
        (B,G,R) = image[100,50]
        print("R={}, G={}, B={}".format(R,G,B))
    #fatiamento e corte da matriz
    def slice(img,cut_x1,cut_x2,cut_y1,cut_y2):
        roi = image[cut_x1:cut_x2,cut_y1:cut_y2]
        cv2.imshow("ROI",roi)
        cv2.waitKey(0)
    def resize(lenght):
        #esta parte considera a proporção adequada
        r = lenght/image.shape[1]
        dim = (300,int(image.shape[0]*r))
        redim = cv2.resize(image,dim)
    def rotated():
        center = (w//2,h//2)
        M = cv2.getRotationMatrix2D(center ,-45,1.0)
        roteted = cv2.warpAffine(image, M,(w,h))
        basic_opencv.show_img('rotação',roteted)

#basic_opencv.show(img)
basic_opencv.segmentation_pixel()
#basic_opencv.slice(img,60,160,0,261)
basic_opencv.resize(300)
#basic_opencv.rotated()
rot = imutils.rotate_bound(image,45)
basic_opencv.show_img('rot',rot)
