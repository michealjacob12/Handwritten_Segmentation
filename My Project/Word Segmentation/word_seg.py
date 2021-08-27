import cv2 
import numpy as np
import os
import sys
import cv2 as cv

def show_wait_destroy(winname, img):    
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.resizeWindow(winname, 600, 600)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)
def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)
    src = cv.detailEnhance(src, sigma_s=10, sigma_r=0.15) 
    height = src.shape[0]
    width = src.shape[1]
    src = cv.resize(src, dsize =(1320, int(1320*height/width)), interpolation = cv.INTER_AREA)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    # Show source image
    show_wait_destroy("Imported Image", src)
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    print('Image processing Started...')
#------------------------------------------------------------VERTICAL LINES----------------------------------------------------------------
    print('Removing Vertical Lines...')
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    # Show gray image
    show_wait_destroy("gray", gray)
    # [gray]
    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    # [bin]
    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    vertical = np.copy(bw)
        
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)
    contours, hierarchy = cv.findContours(vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #mask = np.ones(src.shape[:2], dtype=src.dtype) * 255
    im2 = src.copy()
    color = [255, 255, 255]
    for cnt in contours:
        cv.fillPoly(im2, cnt, color)    
   # if cv.contourArea(cnt) > 100:
        #    x,y,w,h = cv.boundingRect(cnt)
         #   cv.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
          #  cv.drawContours(mask, [cnt], -1, 0, -1)
#        cv.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)
    show_wait_destroy("im2", im2)
    #masked = cv.bitwise_and(src, src, mask=mask)
    #show_wait_destroy("vertical_masked", masked)
    #cv.imwrite("vertical_mask.jpg", masked)
    show_wait_destroy("vertical_lines_removed", im2)
    cv.imwrite("Results/vertical_lines_removed.jpg", im2)
#---------------------------------------------------HORIZONTAL LINE--------------------------------------------------------------------------
    print('Removing Horizontal Lines...')
    img = cv.imread("Results/vertical_lines_removed.jpg", 1)
    # [gray]
    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    # Show binary image
    # [bin]
    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    # [init]
    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    show_wait_destroy("horizontal", horizontal)
    contours, hierarchy = cv.findContours(horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#    conts = imutils.grab_contours(contours)
    im2 = img.copy()
    #mask = np.ones(im2.shape[:2], dtype=im2.dtype) * 255
    #stencil = np.zeros(im2.shape).astype(im2.dtype)
    color = [255, 255, 255]
    for cnt in contours:
        cv.fillPoly(im2, cnt, color)
    show_wait_destroy("Horizontal Lines Removed", im2)
    cv.imwrite("Results/full_lines_removed_image.jpg", im2)
#--------------------------------------------------------------WORD SEGMENTATION---------------------------------------------------------------
    print('Initiating Structuring and Morphing Sequence...')
    print("\n........Program Initiated.......\n")
    img = cv.imread('Results/full_lines_removed_image.jpg', cv.IMREAD_COLOR)
    if len(img.shape) != 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    #gray = cv.blur(gray,(5,5))
    #gray = cv.GaussianBlur(gray,(2,2), 0)
    gray = cv.bilateralFilter(gray,9,75,75)
    print("Applying OTSU THRESHOLD")
    ret,thresh = cv.threshold(gray, 100, 255,cv.THRESH_OTSU|cv.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    dilation = cv2.dilate(thresh,kernel,iterations=4)
    erode = cv2.erode(dilation,kernel,iterations=2)
    final_thr = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    show_wait_destroy("dilation", dilation)
    cv.imwrite("Results/dilation.jpg", dilation)
    show_wait_destroy("erosion", erode)
    cv.imwrite("Results/erosion.jpg", erode)
    show_wait_destroy("Morph_Close", final_thr)
#-------------Word segmenting------------#
    chr_img = src.copy()
    cont_img = src.copy()
    contours, hierarchy = cv2.findContours(final_thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cont_img, contours, -1, (255,0,255), 3)
    show_wait_destroy("contour", cont_img)
    cv.imwrite("Results/contour.jpg", cont_img)
    

    image_number = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cnt = cv2.convexHull(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            ROI = chr_img[y:(y+h), x:(x+w)]
            #ROI = cv2.detailEnhance(ROI, sigma_s=100, sigma_r=0.90)
            show_wait_destroy("ROI", ROI)
            cv2.imwrite("Results/ROI/ROI_{}.png".format(image_number), ROI)
            image_number += 1
            cv2.rectangle(chr_img,(x,y),(x+w,y+h),(0,255,0),2)	
		
    show_wait_destroy("segmented_image", chr_img)
    cv.imwrite("Results/segmented image.jpg", chr_img)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
