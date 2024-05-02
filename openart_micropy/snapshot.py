import sensor,image,pyb,time
import os

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(20)# wait for a moment
clock = time.clock()

def check_dir_access(dir):
    exist = False
    for _dir in os.ilistdir("/sd"):
        print(_dir[0])
        if (dir == _dir[0]):
           exist = True
           break
    if not exist:
        print(f"{dir} is not exist,creating it")
        os.mkdir(f"/sd/{dir}")

save_dir = "O"
sys_dir = "/sd/" + save_dir
check_dir_access(save_dir)

print("wait for user")

print("***********start to capture frames***********")
while(True):
#    name = input("image name:")
    for index in range(100):
        img = sensor.snapshot()
        print(f"saving image{index}")
        sensor.snapshot().save(f"{sys_dir}/{index}.jpg")
