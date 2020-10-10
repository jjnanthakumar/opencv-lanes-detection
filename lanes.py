import cv2 as cv
import numpy as np


# import matplotlib.pylab as plt
__author__ = "jjnanthakumar477@gmail.com"

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_img = cv.bitwise_and(img, mask)
    return masked_img


def drawlines(img, lines):
    img = np.copy(img)
    line_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    img = cv.addWeighted(img, 0.8, line_img, 1, 0.0)
    return img


# img = cv.imread('roads1.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
def process(img):
    height = img.shape[0]
    width = img.shape[1]

    ROI = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    # cv.imshow('Roads', img)
    grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # gblur = cv.GaussianBlur(grey, (9, 9), 0)
    # # bilateralfilter = cv.bilateralFilter(grey, 10,150, 256)
    kernel = np.ones((5, 5), np.float) / 25
    filters = cv.filter2D(grey, -1, kernel)
    medblur = cv.medianBlur(filters, 3)
    edgeimg = cv.Canny(medblur, 200, 220)
    cropped = region_of_interest(edgeimg, np.array([ROI], np.int32))

    houghlines = cv.HoughLinesP(cropped, 2, np.pi / 80, 80, lines=np.array([]), minLineLength=40, maxLineGap=80)
    # for line in houghlines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    final = drawlines(img, houghlines)
    return final


# plt.imshow(final)
# plt.imshow(edgeimg)
# plt.show()

cap = cv.VideoCapture('test.mp4')
out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
while cap.isOpened():
    ret, frame = cap.read()
    k = cv.waitKey(1)  # use & 0xFF in 32 bit
    if k == 27:  # esc key ascii - 27
        break
    image = process(frame)
    # print(frame.shape)
    out.write(image)
    cv.imshow('LaneDetection', image)
cap.release()
cv.destroyAllWindows()

print("Created by "+__author__)
