import os 

# make dirs form A to O in dir named number
origin_dir = "number"
for i in range(15):
    dir = os.path.join(origin_dir,f"lable-{chr(ord('A')+i)}")
    image_name = chr(ord('A')+i)+".jpg"
    image_dir = os.path.join(origin_dir,image_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if os.path.exists(dir) and os.path.exists(image_dir):
        # move A.jpg-O.jpg to lable-A - lable-O dirs
        os.rename(image_dir, os.path.join(dir, image_name))