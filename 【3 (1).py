# 车辆目标跟踪，使用yolov4方法
import cv2
import numpy as np
import time
import math
from object_detection import ObjectDetection  # 导入定义好的目标检测方法


#（1）获取目标检测方法
od = ObjectDetection()


#（2）导入视频
filepath = '747867635-1-192.mp4'
cap = cv2.VideoCapture(filepath)

pTime = 0  # 设置第一帧开始处理的起始时间

count = 0  # 记录帧数

center_points_prev = []  # 存放前一帧检测框的中心点

track_objects = {}  # 存放需要追踪的对象

track_id = 0  # 记录追踪对象的索引，把

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
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)


    #（5）目标追踪
    # 内容：1.追踪目标，2.删除消失了的目标的标记，3.有新目标出现时添加标记
    
    # 只在前两帧图像中比较前后两帧检测到的物体的中心点的距离
    if count <= 2:
        # 当前后两帧的同一目标的中心点移动距离小于规定值，认为是同一物体，对其追踪
        for pt1 in center_points_current:  # 当前帧的中心点
            for pt2 in center_points_prev:  # 前一帧的中心点
                
                # 计算距离，勾股定理
                distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
                
                # 如果距离小于20各像素则认为是同一个目标
                if distance < 20:
                    
                    # 记录当前目标的索引及其当前帧中心点坐标
                    track_objects[track_id] = pt1
                    track_id += 1
    
    
    # 在后续的帧中比较的是，当前帧检测物体的中心点与被追踪目标中心点之间的距离
    else:
        # 由于在循环过程中不能删除字典的元素，先复制一份
        track_objects_new = track_objects.copy()
        
        # 被追踪目标的中心点坐标
        for object_id, pt2 in track_objects_new.items(): 
            
            # 假设在当前帧中，我们在上一帧中跟踪的对象不存在了
            object_exist = False  # 当目标在屏幕上消失后，将其对应的标记消除
            
            # 当前帧检测到的物体的中心点
            for pt1 in center_points_current:  
            
                # 计算两者间的距离
                distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
                
                # 如果两者之间的像素距离小于20， 那么就认为是同一个目标
                if distance < 20:
                    
                    # 更新被追踪目标的前一帧的中心点坐标，等于当前帧中的中心点坐标
                    track_objects[object_id] = pt1
                    
                    # 距离小于20，证明在当前帧中，检测的目标还存在
                    object_exist = True
                    
                    # 在当前帧所有已检测目标的中心点坐标中，删除已经更新过的中心点坐标
                    # 已检测目标中剩余的坐标就是新出现的目标，需要添加标记
                    center_points_current.remove(pt1)
                    continue
                    
            # 如果追踪的对象消失了，删除它的标记
            if object_exist == False:
                track_objects.pop(object_id)
                
                  
        #（6）添加新目标
        for pt in center_points_current:  # 删除更新坐标后剩余的检测到的坐标点
            
            # 给新出现的目标加上标记
            track_objects[track_id] = pt
            track_id += 1
                    
        
                
    #（7）显示出每一帧需要追踪的对象
    for object_id, pt in track_objects.items():
        
        # 在追踪目标的中心点画圈
        cv2.circle(img, pt, 5, (255,0,0), -1)
        # 显示该目标的id
        cv2.putText(img, str(object_id), (pt[0], pt[1]-5), 0, 1, (0,0,255),2)
    
            
    # 打印目标的坐标
    print('tracking objects')
    print(track_objects)
        
    # 打印前一帧的中心点坐标
    print('prevent center points')
    print(center_points_prev)
    
    # 打印当前帧的中心点坐标
    print('current center points')
    print(center_points_current)
    
    
    #（9）查看FPS
    cTime = time.time() #处理完一帧图像的时间
    fps = 1/(cTime-pTime)
    pTime = cTime  #重置起始时间
    
    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
    
    #（10）显示图像，输入窗口名及图像数据
    cv2.namedWindow("img", 0)  #调节显示窗口大小
    cv2.imshow('img', img)    
    
    # 复制当前帧的中心点坐标
    center_points_prev = center_points_current.copy()

    # 每帧滞留20毫秒后消失，ESC键退出
    if cv2.waitKey(0) & 0xFF==1:  # 设置为0代表只显示当前帧
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()



# 为什么屏幕上有的标上了id有的没有
# 离相机越近的车辆速度，在画面上的速度越快，两帧之间的距离就变得较大，这时要合理选择阈值，比较前后两帧之间中心点的距离
