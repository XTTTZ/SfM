import cv2
import numpy as np
import glob

################################################################################

print ('criteria and object points set')

# termination criteria
criteria = (3, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objpoint = np.zeros((9 * 6, 3), np.float32)
objpoint[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# arrays to store object points and image points from all the images

# 3d point in real world space
objpoints = []
# 2d points in image plane
imgpoints = []
################################################################################

print('Load Images')

images = glob.glob('CamfixPic\\CamfixPic_Big\\*.jpg')

for frame in images:

    img1 = cv2.imread(frame)
    imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # find chess board corners
    ret, corners = cv2.findChessboardCorners(imgGray, (9, 6), None)

    # print ret to check if pattern size is set correctly
    print (ret)

    # if found, add object points, image points (after refining them)
    if ret == True:
        # add object points
        objpoints.append(objpoint)
        cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
        # add corners as image points
        imgpoints.append(corners)

        # draw corners
        cv2.drawChessboardCorners(img1, (9, 6), corners, ret)

       # cv2.imshow('Image', img1)#chess
        cv2.waitKey(0)
        cv2.destroyAllWindows()

################################################################################

print ('camera matrix')

ret, camMat, distortCoffs, rotVects, transVects = cv2.calibrateCamera(objpoints, imgpoints, imgGray.shape[::-1], None,None)
################################################################################
#distortCoffs为畸变系数
print('camMat\n',camMat)#打印相机内参数矩阵
print('\nrotVects\n', rotVects)#旋转向量
print('\ntransVects\n', transVects)#平移向量
print ('re-projection error')

meanError = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rotVects[i], transVects[i], camMat, distortCoffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    meanError += error

print ("total error: ", meanError / len(objpoints))


################################################################################

def drawAxis(img, corners, imgpoints):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpoints[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(imgpoints[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(imgpoints[2].ravel()), (0, 0, 255), 5)

    return img


################################################################################

def drawCube(img, corners, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1, 2)

    # draw ground floor in green color
    cv2.drawContours(img, [imgpoints[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpoints[i]), tuple(imgpoints[j]), (255, 0, 0), 3)

    # draw top layer in red color
    cv2.drawContours(img, [imgpoints[4:]], -1, (0, 0, 255), 3)

    return img


################################################################################

print ('pose calculation')

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axisCube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

for frame in glob.glob('CamfixPic\\*.jpg'):

    img2 = cv2.imread(frame)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        # find the rotation and translation vector s.
        _,rotVects, transVects, inliers = cv2.solvePnPRansac(objpoint, corners, camMat, distortCoffs)
        #在OpenCV3的python接口中，发生了一些变化,还会返回一个retval值，这个值一般用不到，所以写需要写'_',见https://blog.csdn.net/c20081052/article/details/89501628
        # project 3D points to image plane
        '''
        imgpoints, jac = cv2.projectPoints(axis, rotVecs, transVecs, camMat, distortCoffs)
        img = drawAxis(img, corners, imgpoints)
        '''

        imgpoints, jac = cv2.projectPoints(axisCube, rotVects, transVects, camMat, distortCoffs)
        img2 = drawCube(img2, corners, imgpoints)

        # cv2.imshow('Image with Pose', img2)#cube

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# shape=img1.shape
# print('shape1=',shape)
# shape=img2.shape
# print('shape2=',shape)
# both = np.hstack((img1,img2))
# cv2.imshow('img1&2',both)#用于并列展示两张图片
# cv2.waitKey(0)
# cv2.destroyAllWindows()
