import cv2
from torchvision import transforms
from PIL import Image
import os 

path = 'image'
for root,dirs,files in os.walk(path):
    print(root,dirs,files)
for file in files:
    img = cv2.imread(os.path.join(path,file))
    img_pil = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.RandomGrayscale(1),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.RandomAffine(degrees=180,
                                fill=(0,0,255)),

    ])
    for i in range(50):
        if not os.path.exists(f"enhance_img/{os.path.splitext(file)[0]}"):
            os.makedirs(f"enhance_img/{os.path.splitext(file)[0]}")
        enhance_img = transform(img_pil)
        enhance_img.save(f"enhance_img/{os.path.splitext(file)[0]}/enhacen_img_{i}.jpg")