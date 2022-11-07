# 车辆目标跟踪
import cv2
import numpy as np
import time
import numpy
#（1）导入视频
filepath = '747867635-1-192.mp4'
cap = cv2.VideoCapture(filepath)

pTime = 0  # 设置第一帧开始处理的起始时间


#（2）处理每一帧图像
while True:
    
    # 接收图片是否导入成功、帧图像
    success, img = cap.read()

    # 查看FPS
    cTime = time.time() #处理完一帧图像的时间
    fps = 1/(cTime-pTime)
    pTime = cTime  #重置起始时间
    
    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
    
    # 显示图像，输入窗口名及图像数据
    cv2.namedWindow("img", 0)  # 窗口大小可调整
    cv2.imshow('img', img)
    if cv2.waitKey(20) & 0xFF==27:  #每帧滞留20毫秒后消失，ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()

