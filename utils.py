import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
  
def enhance_image(img,background,cropped_size = np.array([320,320])):
  # floodfill algorithm
  mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  # mask_2 = np.ones(np.add(background.shape,2)[:2], np.uint8)
  # mask_2[:][449:]
  inrange_white_img = cv2.inRange(background, np.array([200,200,200]), np.array([255,255,255]))
  inrange_img = cv2.inRange(background, np.array([0,120,120]), np.array([130,255,255]))
  flood_img = background.copy()
  flood_deta = (5,5,5)
  cv2.floodFill(flood_img, mask, (0,0), 255, flood_deta, flood_deta)
  contours, _ = cv2.findContours((1-mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours2 , _ = cv2.findContours((255-inrange_img),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours3 , _ = cv2.findContours((255-inrange_white_img),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  center,angle = None,None
  approx_size = None

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
      approx_up_width = np.sqrt(np.dot(approx[0]-approx[1],approx[0]-approx[1])) 
      approx_left_length = np.sqrt(np.dot(approx[1]-approx[2],approx[1]-approx[2]))
      approx_down_width = np.sqrt(np.dot(approx[2]-approx[3],approx[2]-approx[3]))
      approx_right_length = np.sqrt(np.dot(approx[3]-approx[0],approx[3]-approx[0]))
      approx_size = np.multiply([(approx_up_width + approx_down_width),(approx_left_length + approx_right_length)],0.51)
      find_box = True

      crope_center = approx.mean(axis=0)
      noise = np.random.normal(-20, 20, (2,)) 
      crope_center += noise
      crope_start = crope_center - cropped_size//2
      crope_end = crope_center + cropped_size//2
      
      # background = cv2.circle(background, np.int32(crope_start), radius=0, color=(255, 255, 255), thickness=10)
      # background = cv2.circle(background, np.int32(crope_end), radius=0, color=(255, 255, 255), thickness=10)
      # plt.imshow(background),plt.show()
      
      crope_start,crope_end = check_in_range_and_fix(crope_start,crope_end,background.shape[:2])
      crope_start = crope_start.astype(int)
      crope_end = crope_end.astype(int)

      break
  # assert find_box,"No quaters found in the image"
  if not find_box:

    # approx_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
    # cv2.drawContours(approx_mask, contour, -1, 255, 3)
    # contour_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
    # cv2.drawContours(contour_mask, contours , -1, 255, 3)
    # plt.subplot(221),plt.imshow(background),plt.title("baclground")
    # plt.subplot(222),plt.imshow(contour_mask),plt.title("contour_mask")
    # plt.subplot(223),plt.imshow(mask),plt.title("mask")
    # plt.subplot(224),plt.imshow(approx_mask),plt.title("approx_mask")
    # plt.show()
    # plt.clf()

    assert False, "No quaters found in the image"
  # calculate the image size in the realworld snapshot image
  max_side_length = approx_size.max()
  img_side_length = img.shape[0]
  # caltulate resize ratio
  ratio = max_side_length / img_side_length


  # rotate image and create new image mask
  rotated_img = perspective_transform(img.copy(), approx, background.shape[:2])
  # fix number image 0-255 distribution caused mask error
  rotated_mask = perspective_transform(img.copy()+5, approx,background.shape[:2])
  mask = np.equal(rotated_mask, 0)*255

  # # calculate crop size and crop to bg_size
  # target_size = [background.shape[1], background.shape[0]]
  # w_crop_start = int(img.shape[0]//2 - center[0][0])
  # w_crop_end = int(w_crop_start + target_size[0])
  # h_crop_start = int(img.shape[1]//2 - center[0][1])
  # h_crop_end = int(h_crop_start + target_size[1])
  # mask = mask[h_crop_start:h_crop_end, w_crop_start:w_crop_end]
  # rotated_img = rotated_img[h_crop_start:h_crop_end, w_crop_start:w_crop_end]

  # bit wise and 
  background = cv2.convertScaleAbs(background)
  mask = cv2.convertScaleAbs(mask)
  masked_img = cv2.bitwise_and(background, mask)
  final_img = cv2.add(rotated_img, masked_img)
  final_img = final_img[crope_start[1]:crope_end[1],crope_start[0]:crope_end[0]]
  if final_img.shape[0] < 320 or final_img.shape[1] < 320:
    assert False, "The image is too small"
    
  # approx_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  # cv2.drawContours(approx_mask, contour, -1, 255, 3)
  # contour_mask = np.zeros(np.add(background.shape,2)[:2], np.uint8)
  # cv2.drawContours(contour_mask, contours , -1, 255, 3)
  # plt.subplot(221),plt.imshow(rotated_mask),plt.title("rotated_mask")
  # plt.subplot(222),plt.imshow(background),plt.title("background")
  # plt.subplot(223),plt.imshow(approx_mask),plt.title("approx_mask")
  # plt.subplot(224),plt.imshow(final_img),plt.title("final_img")
  # plt.show()
  # plt.clf()
  
  return final_img