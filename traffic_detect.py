import cv2
import numpy as np

cap = cv2.VideoCapture('videos/trafficcamera.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

frame_count = 0
while True:
    print(frame_count)
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 20 != 0:
        continue
    img = fgbg.apply(frame)
    print(frame.shape)

    #	img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    cv2.imshow("thresholded", thresh)
    image, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cv2.imshow("contours", image)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    inds = sorted(range(len(areas)), key=lambda k: areas[k])
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for j in range(min(10, len(areas))):
        cnt = contours[inds[-j]]
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            hull = cv2.convexHull(cnt)
            for i in range(len(hull)):
                p = hull[i][0]
                p_next = hull[(i + 1) % len(hull)][0]
                print(p)
                cv2.line(frame, (p[0], p[1]), (p_next[0], p_next[1]), (255, 0, 0), 5)

            print(hull)

        #	print((cx,cy))
        #	cv2.circle(img_color,(cx,cy),5, (0,0,255))
    cv2.imshow("original", frame)

    # for cnt in contours:
    #	area = cv2.contourArea(cnt)
    #	print(area)

    cv2.waitKey(50)
