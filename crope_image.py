import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom
# def rotate_image(image, angle,ratio):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   random_angle = angle + np.random.choice(np.arange(-180,181,step=90),size=1).item()
#   rot_mat = cv2.getRotationMatrix2D(image_center, random_angle, ratio)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

def check_in_range_and_fix(crope_start,crope_end,bg_size):

  bg_size = bg_size[::-1]

  if crope_start[0] < 0:
    crope_end[0] -= crope_start[0]
    crope_start[0] = 0

  if crope_start[1] < 0:
    crope_end[1] -= crope_start[1]
    crope_start[1] = 0

  if crope_end[0] > bg_size[0]:
    crope_start[0] -=  np.abs(crope_end[0] - bg_size[0])
    crope_end[0] = bg_size[0]

  if crope_end[1] > bg_size[1]:
    crope_start[1] -= np.abs(crope_end[1] - bg_size[1])
    crope_end[1] = bg_size[1]

  return crope_start,crope_end

def demo(lst, k):
  x = lst[k-1::-1]
  y = lst[:k-1:-1]
  return np.float32(np.concatenate((x,y),axis=0))

def perspective_transform(image,approx,bg_size):
  approx = demo(approx,np.random.choice([0,1,2,3]).item())
  
  enlage_deta = 40
  origin = np.float32([[enlage_deta,enlage_deta],[enlage_deta,image.shape[1]-enlage_deta],
                       [image.shape[0]-enlage_deta,image.shape[1]-enlage_deta],[image.shape[0]-enlage_deta,enlage_deta],])
  # approx = np.float32(demo(approx,-2))
  transfor_mat = cv2.getPerspectiveTransform(origin,approx)
  out = cv2.warpPerspective(image, transfor_mat, (image.shape[1],image.shape[1]))
  out = out[0:bg_size[0],0:bg_size[1]]
  return out

def get_approx_angle(approx) -> list[np.array,np.array]:
  # oblique in the image
  center = np.mean(approx, axis= 0)
  mid_x  = np.mean(approx[:][:2],axis=0)
  mid_y  = np.mean(approx[:][2:],axis=0)

  # using array to calculate angle
  array_a = np.array([1,0])
  array_b = np.squeeze(mid_x - mid_y)
  angle = math.degrees(math.acos(
      np.dot(array_a,array_b)/(np.linalg.norm(array_a)*np.linalg.norm(array_b))))
  
  return center, angle

def restore_to_original_size(cropped_img, original_height, original_width):
    # 计算缩放比例
    scale_height = original_height / cropped_img.shape[0]
    scale_width = original_width / cropped_img.shape[1]
    
    # 使用zoom函数进行插值放大
    restored_img = zoom(cropped_img, (scale_height, scale_width, *([1] * (cropped_img.ndim - 2))))
    
    return restored_img

def initial_crope_image(img,ratio):
    # 获取原始图像的宽度和高度
    original_height, original_width = img.shape[:2]
    
    # 计算裁剪后的宽度和高度
    new_width = int(original_width * ratio)
    new_height = new_width 
    # 根据新的尺寸裁剪图像
    # 假设我们从图像的中心开始裁剪
    start_row = (original_height - new_height) // 2
    start_col = (original_width - new_width) // 2
    img = img[start_row:start_row + new_height, start_col:start_col + new_width]
    # 返回裁剪后的图像

    return img

def crope_image(img:np.ndarray,cropped_size = np.array([128,128])):
  # floodfill algorithm
  
  img = initial_crope_image(img,0.6)
  mask = np.zeros(np.add(img.shape,2)[:2], np.uint8)
  # mask_2 = np.ones(np.add(background.shape,2)[:2], np.uint8)
  # mask_2[:][449:]
  inrange_white_img = cv2.inRange(img, np.array([200,200,200]), np.array([255,255,255]))
  inrange_img = cv2.inRange(img, np.array([0,120,120]), np.array([130,255,255]))
  flood_img = img.copy()
  flood_deta = (5,5,5)
  cv2.floodFill(flood_img, mask, (0,0), 255, flood_deta, flood_deta)
  contours, _ = cv2.findContours((1-mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours2 , _ = cv2.findContours((255-inrange_img),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours3 , _ = cv2.findContours((255-inrange_white_img),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  find_box = False
  for contour in contours3 + contours2+contours:
    # calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)


    # get the position and angle of image
    if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 10000 and cv2.contourArea(approx) < 50000:
      # mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
      # cv2.drawContours(mask,[approx] , -1, 255, 3)
      # plt.imshow(mask),plt.show()

      approx = np.squeeze(approx)
      find_box = True

      crope_center = approx.mean(axis=0)
      crope_start = crope_center - cropped_size//2
      crope_end = crope_center + cropped_size//2
      
      # crope_start,crope_end = check_in_range_and_fix(crope_start,crope_end,img.shape[:2])
      crope_start = crope_start.astype(int)
      crope_end = crope_end.astype(int)

      break

  if not find_box:
    background= img
    approx_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
    cv2.drawContours(approx_mask, contour, -1, 255, 3)
    contour_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
    cv2.drawContours(contour_mask, contours , -1, 255, 3)
    plt.subplot(221),plt.imshow(background),plt.title("baclground")
    plt.subplot(222),plt.imshow(contour_mask),plt.title("contour_mask")
    plt.subplot(223),plt.imshow(mask),plt.title("mask")
    plt.subplot(224),plt.imshow(approx_mask),plt.title("approx_mask")
    plt.show()
    plt.clf()

    assert False, "No quaters found in the image"
  # calculate the image size in the realworld snapshot image

  final_img = img[crope_start[1]:crope_end[1],crope_start[0]:crope_end[0]]
  if final_img.shape[0] < cropped_size[0] or final_img.shape[1] < cropped_size[1]:
    assert False, "The image is too small"
    
  background = img
  approx_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  cv2.drawContours(approx_mask, [approx], -1, 255, 3)
  contour_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  cv2.drawContours(contour_mask, contours , -1, 255, 3)
  plt.subplot(221),plt.imshow(img),plt.title("img")
  plt.subplot(222),plt.imshow(background),plt.title("background")
  plt.subplot(223),plt.imshow(approx_mask),plt.title("approx_mask")
  plt.subplot(224),plt.imshow(final_img),plt.title("final_img")
  plt.show()
  plt.clf()
  
  return final_img