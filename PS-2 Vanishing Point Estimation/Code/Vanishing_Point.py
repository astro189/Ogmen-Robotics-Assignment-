import cv2
import numpy as np


def detect_edges(image):
    """
    Function to detect the edges of the image

    Input: Image 
    Output: Edges
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3), 1)
    edges = cv2.Canny(blur,100, 255)

    return edges

def detect_lines(edges):
    """
    Function to detect the lines in the image

    Input: Edges 
    Output: Lines
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 10, 15)
    return lines

def filter_lines(lines, maxCount, AngleThresh):

    """
    Function to filter out too small or completely horizontal and vertical lines

    Input: lines = Lines data, maxCount = Maximum lines to keep, AngleThresh =  angle threshold to consider
    Output: filtered lines
    """
     
    filtered_lines = []


    for line in lines :
        x1, y1, x2, y2 = line[0]

        #Compute slope
        if x1!=x2:
            m = (y2-y1)/(x2-x1)

        else:
          m =  np.float16('inf')
        
        theta = np.rad2deg(np.arctan(m))

        #Only if slop in expected range store it 
        if AngleThresh<= abs(theta) <= (90 - AngleThresh):
            c = y2 - x2*m
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            filtered_lines.append([line[0], length, c, m])

    filtered_lines =  sorted(filtered_lines, key=lambda x:x[1], reverse=True)
    filtered_lines = filtered_lines[:maxCount+1]

    return filtered_lines

def find_vanishing_point(lines):
    """
    Function to find the vanishing point 

    Input: Filtered lines
    Output: Vanishing point 
    """
 
    min_error = float('inf')

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):

            #Compute intersection point (x0, y0) between a pair of lines
            m1, c1, m2, c2 = lines[i][3], lines[i][2], lines[j][3], lines[j][2]

            if m1!=m2:
                x_intersection = (c2 - c1)/(m1 - m2)
                y_intersection = x_intersection*m1 + c1
            
            error = 0
            for k in range(len(lines)):
                    #Compute intersection point between each line and its corresponding perpendicular (x', y')
                    Lm, Lc = lines[k][3], lines[k][2]

                    m_perpendicular = -1 / Lm
                    c_perpendicular = y_intersection - m_perpendicular * x_intersection
        
                    x_perpendicular_intersection = (Lc - c_perpendicular) / (m_perpendicular - Lm)
                    y_perpendicular_intersection = m_perpendicular * x_perpendicular_intersection + c_perpendicular

                    #Find distance between (x0, y0) and (x', y') and add to error
                    distance = np.sqrt((y_perpendicular_intersection - y_intersection) ** 2 +(x_perpendicular_intersection - x_intersection) ** 2)
                    error += distance ** 2  

            error = np.sqrt(error)
            
            #Intersection with minimum error is vanishing point 
            if error < min_error:
                min_error = error
                best_vanishing_point = (x_intersection, y_intersection)

        return best_vanishing_point
    

img = cv2.imread(r'sample2.jpg')
edges = detect_edges(img)
lines= detect_lines(edges)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
filtered_lines = filter_lines(lines, maxCount=15, AngleThresh=4)


# for values in filtered_lines:
#     line = values[0]
#     cv2.line(edges, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0), 3)

VanishingPoint = find_vanishing_point(filtered_lines)

cv2.circle(img, (int(VanishingPoint[0]), int(VanishingPoint[1])),5, color=(0, 0, 255), thickness=cv2.FILLED)

if img.shape[0]>800 or img.shape[1]>800:
    img = cv2.resize(img, dsize=None, fx=0.4, fy=0.4)

cv2.imwrite(r'vanishing_point.png', img)
cv2.imshow('Vanishing Point', img)
cv2.waitKey(0)