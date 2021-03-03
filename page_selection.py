import os
import argparse 
import numpy as np
import cv2




class quadrilateral:

    """
            c1    mp1   c2
                *----*----*
                -----------
            mp4 *---------* mp2
                -----------
                *----*----*
            c4    mp3   c3

    """

    def __init__(self,pagecontourpoints, r=5, border_color=(255, 0, 0),
                  corner_color=(0, 0, 255)):
        # Initialize the contours
        self.contours = np.array([[pagecontourpoints[0,0] + r, pagecontourpoints[0,1] + r], # c1
                               [pagecontourpoints[1,0] + r, pagecontourpoints[1,1] - r],    # c2
                               [pagecontourpoints[2,0] - r, pagecontourpoints[2,1] - r],    # c3
                               [pagecontourpoints[3,0] - r, pagecontourpoints[3,1] + r]])   # c4


        # get mp1,mp2,mp3 and mp4
        self.middlepoints=self.get_middlepoints()
        
        self.slope = np.array([((self.contours[(i+1)%4,1]-self.contours[i,1])/(self.contours[(i+1)%4,0]-self.contours[i,0]))
                            for i in range(len(self.contours))]).astype("float32")

        
        # Initialize the radius of the corners
        self.r = r
        # Initialize the colors of the quadrilateral

        self.border_color = border_color
        self.corner_color = corner_color

    def get_middlepoints(self):
        return np.array([((self.contours[i,0]+self.contours[(i+1)%4,0])//2,
                                (self.contours[i,1]+self.contours[(i+1)%4,1])//2)
                                for i in range(len(self.contours))]).astype("int")

    def get_corner_index(self, coord):
        # A corner is return if the coordinates are in its radius
        for i, b in enumerate(self.contours):
            dist = sum([(b[i] - x) ** 2 for i, x in enumerate(coord)]) ** 0.5
            if  dist < self.r:
                return i
        # If no corner, return None
        return None

    def get_middlepoint_index(self, coord):
        # A middlepoint is return if the coordinates are in its radius


        for i, b in enumerate(self.middlepoints):
            dist = sum([(b[i] - x) ** 2 for i, x in enumerate(coord)]) ** 0.5
            if  dist < self.r:
                return i
        # If no middlepoint, return None
        return None



    def update_corner_and_middlepoint(self, corner_index,middlepoint_index, coord):

         if corner_index is not None:
            self.contours[corner_index] = coord

            self.middlepoints=self.get_middlepoints()

         if middlepoint_index is not None:
            i=middlepoint_index
            c=self.contours
            s=self.slope
            (x,y)=coord

            
            self.contours[i] = (round((c[i,1] - y +s[i]*x -s[i-1]*c[i,0])/(s[i] - s[i-1])),
                                round((s[i]*c[i,1] - s[i-1]*s[i]*c[i,0] + s[i-1]*s[i]*x - s[i-1]*y)/(s[i] - s[i-1]))
                                )

            self.contours[(i+1)%4] = (round((c[(i+1)%4,1] - y + s[i]*x - s[(i+1)%4]*c[(i+1)%4,0])/(s[i]-s[(i+1)%4])),
                                      round((s[i]*c[(i+1)%4,1] - s[(i+1)%4]*s[i]*c[(i+1)%4,0] + s[(i+1)%4]*s[i]*x - s[(i+1)%4]*y )/(s[i] - s[(i+1)%4]))
                                      )


            self.middlepoints=self.get_middlepoints() 

            

          
   


class Scanner():

    def __init__(self, input_path, output_path):
        self.input = cv2.imread(input_path)
        self.shape = self.input.shape[:-1] #height*width
        self.size = tuple(list(self.shape)[::-1])
        self.output_path = output_path
        self.pagecontourpoints=pagecontour(self.input)

        # create a quadrilateral to drag and drop and its perspective matrix
        self.M = None
        self.quadrilateral = quadrilateral(self.pagecontourpoints,
                                
                                r=min(self.shape) // 100 ,
                                border_color=(255, 0,0 ),
                                corner_color=(0, 0,255))

        # Initialize the opencv window
        cv2.namedWindow('Rendering', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Rendering', self.drag_and_drop_border)

        # to remember wich border is dragged if exists
        self.corner_dragged = None
        self.middlepoint_dragged = None


    def draw_quadrilateral(self, img):
        # draw the contours of the quadrilateral
        cv2.drawContours(img, [self.quadrilateral.contours], -1,
                       self.quadrilateral.border_color, self.quadrilateral.r//3)

        # Draw the corner of the trapezoid as circles
        for x, y in self.quadrilateral.contours:
            cv2.circle(img, (x, y), self.quadrilateral.r,
                      self.quadrilateral.corner_color, cv2.FILLED)

        for x, y in self.quadrilateral.middlepoints:
            cv2.circle(img, (x, y), self.quadrilateral.r,
                      self.quadrilateral.corner_color, cv2.FILLED)

        return img


    def drag_and_drop_border(self, event, x, y, flags, param):
        # If the left click is pressed, get the point to drag
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the selected point if exists
            self.corner_dragged = self.quadrilateral.get_corner_index((x, y))
            self.middlepoint_dragged = self.quadrilateral.get_middlepoint_index((x, y))

        # If the mouse is moving while dragging a point, set its new position
        elif event == cv2.EVENT_MOUSEMOVE and self.corner_dragged is not None:
                self.quadrilateral.update_corner_and_middlepoint(self.corner_dragged,None,(x, y))

        elif event == cv2.EVENT_MOUSEMOVE and self.middlepoint_dragged is not None:
                self.quadrilateral.update_corner_and_middlepoint(None,self.middlepoint_dragged,(x, y))


        # If the left click is released
        elif event == cv2.EVENT_LBUTTONUP:
            # Remove from memory the selected point
  
            self.corner_dragged = None
            self.middlepoint_dragged = None


    def actualize_perspective_matrices(self):
        # get the source points (quadrilateral)
        src_pts = self.quadrilateral.contours.astype(np.float32)

        # set the destination points to have the perspective output image
        w, h = self.shape
        dst_pts = np.array([[0, 0],
                          [h-1, 0],
                          [h - 1,  w- 1],
                          [0, w-1]], dtype="float32")
 
        # compute the perspective transform matrices
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
 

    def run(self):
        while True:


            # draw current state of the quadrilateral
            img_input = self.draw_quadrilateral(self.input.copy())
            # Display until the 'Enter' key is pressed
            cv2.imshow('Rendering', img_input)
            if cv2.waitKey(1) & 0xFF == 13:
                break

        self.actualize_perspective_matrices()
        # get the output image according to the perspective transformation
        img_output = cv2.warpPerspective(self.input, self.M, self.size)
        # Save the image and exit the process
        cv2.imwrite(self.output_path,img_output)
        cv2.destroyAllWindows()


scanner = Scanner("img/1.jpg", "img/1_scanned.jpg")
scanner.run()