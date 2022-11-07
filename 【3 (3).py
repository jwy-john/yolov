# 车辆目标跟踪，使用yolov4方法
import cv2
import numpy as np
import time
from object_detection import ObjectDetection  # 导入定义好的目标检测方法


#（1）获取目标检测方法
od = ObjectDetection()


#（2）导入视频
filepath = '747867635-1-192.mp4'
cap = cv2.VideoCapture(filepath)

pTime = 0  # 设置第一帧开始处理的起始时间

count = 0  # 记录帧数

center_points_prev = []  # 存放前一帧检测框的中心点

#（3）处理每一帧图像
while True:
    
    count += 1  # 记录当前是第几帧
    print('------------------------')
    print('NUM:', count)
    
    # 接收图片是否导入成功、帧图像
    success, img = cap.read()
    
    # 如果读入不到图像就退出
    if success == False:
        break
    
    center_points_current = []  # 储存当前帧的所有目标的中心点坐标
    
    
    #（4）目标检测
    # 将每一帧的图像传给目标检测方法
    # 返回class_ids图像属于哪个分类；scores图像属于某个分类的概率；boxes目标检测的识别框
    class_ids, scores, boxes  = od.detect(img)
    
    # 绘制检测框，boxes中包含每个目标检测框的左上坐标和每个框的宽、高
    for box in boxes:
        (x, y, w, h) = box
        
        # 获取每一个框的中心点坐标，像素坐标是整数
        cx, cy = int((x+x+w)/2), int((y+y+h)/2) 
        
        # 存放每一帧的所有框的中心点坐标
        center_points_current.append((cx,cy))
        
        # 绘制矩形框。传入帧图像，框的左上和右下坐标，框颜色，框的粗细
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    
    # 显示所有检测框的中心点，pt代表所有中心点坐标
    for pt in center_points_current:
        cv2.circle(img, pt, 5, (0,0,255), -1)
        
    # 打印前一帧的中心点坐标
    print('prevent center points')
    print(center_points_prev)
    
    # 打印当前帧的中心点坐标
    print('current center points')
    print(center_points_current)       
    
    # 查看FPS
    cTime = time.time() #处理完一帧图像的时间
    fps = 1/(cTime-pTime)
    pTime = cTime  #重置起始时间
    
    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
    
    # 显示图像，输入窗口名及图像数据
    cv2.imshow('img', img)    
    
    # 复制当前帧的中心点坐标
    center_points_prev = center_points_current.copy()

    # 每帧滞留20毫秒后消失，ESC键退出
    if cv2.waitKey(30) & 0xFF==27:  # 设置为0代表只显示当前帧
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()
