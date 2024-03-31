import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def rotate_image(image, angle,ratio):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, ratio)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

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
  
def enhance_image(img,background):
  # floodfill algorithm
  mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  flood_img = background.copy()
  flood_deta = (5,5,5)
  cv2.floodFill(flood_img, mask, (0,0), 255, flood_deta, flood_deta)
  contours, _ = cv2.findContours((1-mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  center,angle = None,None
  approx_size = None

  find_box = False
  for contour in contours:
    # calculate perimeter
    # mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
    # cv2.drawContours(mask, contour, -1, 255, 3)
    # plt.imshow(mask),plt.show()
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # get the position and angle of image
    if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 1000 and cv2.contourArea(approx) < 30000:
      center, angle = get_approx_angle(approx)

      approx = np.squeeze(approx)
      approx_up_width = np.sqrt(np.dot(approx[0]-approx[1],approx[0]-approx[1])) 
      approx_left_length = np.sqrt(np.dot(approx[1]-approx[2],approx[1]-approx[2]))
      approx_down_width = np.sqrt(np.dot(approx[2]-approx[3],approx[2]-approx[3]))
      approx_right_length = np.sqrt(np.dot(approx[3]-approx[0],approx[3]-approx[0]))
      approx_size = np.multiply([(approx_up_width + approx_down_width),(approx_left_length + approx_right_length)],0.6)
      find_box = True
      break
  assert find_box,"No quaters found in the image"
  # calculate the image size in the realworld snapshot image

  # caltulate resize ratio
  max_side_length = approx_size.max()
  img_side_length = img.shape[0]
  ratio = max_side_length / img_side_length


  # rotate image and create new image mask
  rotated_img = rotate_image(img, angle, ratio)
  mask = np.equal(rotated_img, 0)*255

  # calculate crop size and crop
  target_size = [background.shape[1], background.shape[0]]
  w_crop_start = int(img.shape[0]//2 - center[0][0])
  w_crop_end = int(w_crop_start + target_size[0])
  h_crop_start = int(img.shape[1]//2 - center[0][1])
  h_crop_end = int(h_crop_start + target_size[1])
  mask = mask[h_crop_start:h_crop_end, w_crop_start:w_crop_end]
  rotated_img = rotated_img[h_crop_start:h_crop_end, w_crop_start:w_crop_end]

  # bit wise and 
  background = cv2.convertScaleAbs(background)
  mask = cv2.convertScaleAbs(mask)
  masked_img = cv2.bitwise_and(background, mask)
  final_img = cv2.add(rotated_img, masked_img)

  return final_img