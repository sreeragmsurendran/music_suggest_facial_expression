import cv2

run=False
frame=0
#path=#some video path2

def foo(event, x, y, flags, param):
    global run
    global frame
    #check which mouse button was pressed
    #e.g. play video on left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        run= not run
        while run:

            frame+=1
            frame=cap.read()[1]
            cv2.imshow(window_name, frame)


window_name='videoPlayer'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, foo)

cap=cv2.VideoCapture(0)