import sensor, image, time,pyb,gc,tf,math
import time
from pyb import LED
from machine import UART
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.B128X128)
sensor.skip_frames(time = 2000)
uart = UART(2, baudrate=115200)
net_path = "number.tflite"
labels = [line.rstrip() for line in open("/sd/number.txt")]
net = tf.load(net_path, load_to_fb=True)
leibie = 0
rect_map= [0,0,128,128]
while(True):
    img = sensor.snapshot()
    img.lens_corr(1.5)

    for obj in tf.classify(net ,img, roi=rect_map,min_scale=1.0, scale_mul=0.5, x_overlap=0.5, y_overlap=0.5):
        # img.draw_rectangle(rect_map, color = (0, 0, 0),thickness = 4)   # 绘制矩形外框，便于在IDE上查看识别到的矩形位置
        #print("**********\nTop 1 Detections at net [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
        find_obj = sorted_list[0][0]

        if find_obj == 'qiezi' or find_obj == 'putao' or find_obj == 'yumi':
            leibie = 5
            print(leibie)
        elif find_obj == 'huasheng' or find_obj == 'pingguo' or find_obj == 'lajiao':
            leibie = 3
        elif find_obj == 'chandou' or find_obj == 'chengzi' or find_obj == 'baicai':
            leibie = 4
        elif find_obj == 'fanshu' or find_obj == 'liulian' or find_obj == 'huanggua':
            leibie = 2
        elif find_obj == 'shuidao' or find_obj == 'xiangjiao' or find_obj == 'luobo':
            leibie = 1
        img.draw_rectangle(120,40,430,255, color = (0, 0, 0),thickness = 4)
        shibie = sorted_list[0][0]
        shibie_str = 'i'+str(leibie)+'h'
        uart.write(shibie_str,12)
        print(sorted_list)
    sorted_list.clear()
    gc.collect()





