import sensor,image,pyb,time
import os

RED_LED_PIN = 1
BLUE_LED_PIN = 3

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(20)# wait for a moment
clock = time.clock()

def check_dir_access(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else :
        pass

save_dir = "/sd/three"
print("wait for user")

print("***********start to capture frames***********")
while(True):
#    name = input("image name:")
    for index in range(100):
        img = sensor.snapshot()
        print(f"saving image{index}")
        check_dir_access(save_dir)
        sensor.snapshot().save(f"{save_dir}/{index}.jpg")