import numpy as np
import time
import cv2, math
import cv2.aruco as aruco
img = cv2.imread('239.JPG')

def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:  # 偏航，俯仰，滚动
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # 偏航，俯仰，滚动换成角度
    rx = x * 180.0 / 3.141592653589793
    ry = y * 180.0 / 3.141592653589793
    rz = z * 180.0 / 3.141592653589793
    return rx, ry, rz

mtx = np.array([
    [2946.48, 0, 1980.53],
    [0, 2945.41, 1129.25],
    [0, 0, 1],
])
dist = np.array([0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892])

#mtx = np.array([[629.61554535, 0.      , 333.57279485],[0.      , 631.61712266, 229.33660831],[ 0.        , 0.        , 1.        ]])
#dist = np.array(([[0.03109901, -0.0100412, -0.00944869, 0.00123176, 0.31024847]]))
#cap = cv2.VideoCapture('Video.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
#num = 0
while True:
    #ret, frame = cap.read()
    h1, w1 = img.shape[:2]
    # print(h1, w1)
    # 读取摄像头画面
    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    img = dst1
    # print(newcameramtx)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters_create()
    dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
    '''
    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''

    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#    如果找不打id
    if ids is not None:

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.035, mtx, dist)
        # 估计每个标记的姿态并返回nt(值rvet和tvec ---不同
        # from camera coeficcients
        (rvec-tvec).any()# get rid of that nasty numpy value array error
        for i in range(rvec.shape[0]):
            aruco.drawAxis(img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(img, corners)
        imgpts = [np.int32(corners).reshape(-1, 2)]
        for pt in imgpts[0]:
            cv2.drawMarker(img, position=tuple(pt), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=30,
                           thickness=3)
        ###### DRAW ID #####
        cv2.putText(img, "Id: " + str(ids), (0, 40), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        EulerAngles = rotationVectorToEulerAngles(rvec)
        EulerAngles = [round(i, 2) for i in EulerAngles]
        cv2.putText(img, "Attitude_angle:" + str(EulerAngles), (0, 220), font, 1.5, (0, 255, 0), 3,
                    cv2.LINE_AA)
        tvec = tvec * 1000
        for i in range(3):
            tvec[0][0][i] = round(tvec[0][0][i], 1)
        tvec = np.squeeze(tvec)
        cv2.putText(img, "Position_coordinates:" + str(tvec) + str('mm'), (0, 120), font, 1.5, (0, 255, 0), 3,
                    cv2.LINE_AA)
    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(img, "No Ids", (0, 40), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.namedWindow('frame', 0)
    # cv2.resizeWindow("frame", 960, 720)
    # 显示结果框架
    #img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("img", img)

    key = cv2.waitKey(0)

