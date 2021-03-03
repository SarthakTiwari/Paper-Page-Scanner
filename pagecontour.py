import numpy as np
import cv2


def resize(img, height=800):
    # Resize image to given height (to speed up process)
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

def fourCornersSort(pts):
    # Sort corners: top-left,top-right,bot-right, bot-left

    # Difference and sum of x and y value
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmin(diff)],
                     pts[np.argmax(summ)],
                    pts[np.argmax(diff)] ])

def contourOffset(cnt, offset):
    # Offset contour, by 5px border

    # Matrix addition
    cnt += offset
    
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


def pagecontour(input_image):

    # Load image and convert it from BGR to RGB
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Resize and convert to grayscale
    img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)

    # Bilateral filter to preserve edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # blurring to clear small details
    img = cv2.medianBlur(img, 11)

    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    edges = cv2.Canny(img, 200, 250)



    # Getting contours  
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding contour of biggest rectangle
    # Otherwise return corners of original image

    height = edges.shape[0]
    width = edges.shape[1]
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10) # 5px border considered 

    # Page fill at least half of image, then saving max area found
    maxAreaFound = MAX_COUNTOUR_AREA * 0.5

    # Saving page contour
    pageContour = np.array([[[5, 5]], [[5, height-5]], [[width-5, height-5]], [[width-5, 5]]])

    # Go through all contours
    for cnt in contours:
        # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound 
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx


    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])
    pageContour = contourOffset(pageContour, (-5, -5))
    sPoints = pageContour.dot(image.shape[0] / 800)
    return sPoints.astype(int)
