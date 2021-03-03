import os
import argparse 
import numpy as np
import cv2
from pagecontour import pagecontour




class quadrilateral:

    """
            c0    mp0   c1
                *----*----*
                -----------
            mp3 *---------* mp1
                -----------
                *----*----*
            c3    mp2   c2

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
        
        # get slope of edges
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

            '''
                for parellel shift of edges

                            (x1_new,y1_new)
                                ^                                 slope :- 
                    (x0,y0)  ___|_________(x1,y1)                   s0 = (y1-y0)/(x1-x0) 
                            |   :        |                                 .
                            |   :        |                                 .
                            |   *(x,y)   * middle point 1               and so on 
                            |   :        |
                            |___ ________|
                    (x3,y3)     |          (x2,y2)
                                V
                            (x2_new,y2_new)

                when middle point 1 is shifted to (x,y)

                (x1_new,y1_new) = (
                                   (y1 - y + s1*x - s0*x1)/(s1 - s0),
                                   (s1*y1 - s0*s1*x1 + s0*s1*x - s0*y)/(s1 - s0)
                                  )

                (x2_new,y2_new) = (
                                   (y2 - y + s1*x - s2*x2)/(s1 - s2),
                                   (s1*y2 - s2*s1*s2 + s2*s1*x - s2*y)/(s1 - s2)
                                  )
            '''

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

        # Draw the corner of the quadrilateral as circles
        for x, y in self.quadrilateral.contours:
            cv2.circle(img, (x, y), self.quadrilateral.r,
                      self.quadrilateral.corner_color, cv2.FILLED)

        # Draw the middlepoints of the quadrilateral as circles
        for x, y in self.quadrilateral.middlepoints:
            cv2.circle(img, (x, y), self.quadrilateral.r,
                      self.quadrilateral.corner_color, cv2.FILLED)

        return img


    def drag_and_drop_border(self, event, x, y, flags, param):
        # If the left click is pressed, get the middlepoint or corner to drag
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




if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Paper Page Detector and selector ")
    parser.add_argument("-in","--input_image",type=str,help="Input image :- .../image_name.format")
    parser.add_argument("-out","--output_path",type=str,help="Output image path:- .../desired_image_name.desired_format ",default=None)

    args = parser.parse_args()

    if args.input_image is None:
        print("No image specified")

    elif not os.path.exists(args.input_image):
        print('The input image specified path does not exists')

    else:
        if args.output_path is None:
           path,img_format=args.input_image.split(".")
           outpath= path+"_scanned."+img_format
        else:
            outpath=args.output_path

        scanner = Scanner(args.input_image, outpath)
        scanner.run()