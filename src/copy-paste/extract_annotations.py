import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

imagesRoot = "../../data/coco_minitrain_25k/images_pruned/giraffes_elephants"
annotationsRoot = "../../data/coco_minitrain_25k/annotations"

# elephant_dict = {}
#
# for elephant in tqdm(os.listdir("../../data/coco_minitrain_25k/images_pruned/giraffes_elephants/elephant")):
#     path = os.path.join(imagesRoot+'/elephant', elephant)
#
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     ### from here: https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
#     # create a simple mask image similar
#     # to the loaded image, with the
#     # shape and return type
#     mask = np.zeros(image.shape[:2], np.uint8)
#
#     # specify the background and foreground model
#     # using numpy the array is constructed of 1 row
#     # and 65 columns, and all array elements are 0
#     # Data type for the array is np.float64 (default)
#     backgroundModel = np.zeros((1, 65), np.float64)
#     foregroundModel = np.zeros((1, 65), np.float64)
#
#     # define the Region of Interest (ROI)
#     # as the coordinates of the rectangle
#     # where the values are entered as
#     # (startingPoint_x, startingPoint_y, width, height)
#     # these coordinates are according to the input image
#     # it may vary for different images
#     w, h = np.shape(image)[0], np.shape(image)[1]
#
#     rectangle = (0, 0, w - 1, h - 1)
#
#     # apply the grabcut algorithm with appropriate
#     # values as parameters, number of iterations = 3
#     # cv2.GC_INIT_WITH_RECT is used because
#     # of the rectangle mode is used
#     cv2.grabCut(image, mask, rectangle,
#                 backgroundModel, foregroundModel,
#                 5, cv2.GC_INIT_WITH_RECT)  # iteration number seems to be an important parameter
#
#     # In the new mask image, pixels will
#     # be marked with four flags
#     # four flags denote the background / foreground
#     # mask is changed, all the 0 and 2 pixels
#     # are converted to the background
#     # mask is changed, all the 1 and 3 pixels
#     # are now the part of the foreground
#     # the return type is also mentioned,
#     # this gives us the final mask
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#
#     # The final mask is multiplied with
#     # the input image to give the segmented image.
#     image = image * mask2[:, :, np.newaxis]
#
#     mask_inv = cv2.bitwise_not(mask2)
#
#     ### from here: https://stackoverflow.com/questions/72408809/how-do-i-fill-up-mask-holes-in-opencv
#     # find the largest contour
#     contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
#
#     # draw the largest contour to fill in the holes in the mask
#     final_result = np.ones(image.shape[:2])  # create a blank canvas to draw the final result
#     final_result = cv2.drawContours(final_result, [largest_contour], -1, color=(0, 255, 0), thickness=cv2.FILLED)
#
#     # show results
#     final_mask = final_result.astype('uint8')
#     final_mask = np.logical_not(final_mask).astype('uint8')
#
#     # Get the bbox (xmin, ymin, width, height)
#     bbox = 0, 0, 0, 0
#     segmentation = np.where(final_mask == 1)
#
#     if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
#         x_min = int(np.min(segmentation[1]))
#         x_max = int(np.max(segmentation[1]))
#         y_min = int(np.min(segmentation[0]))
#         y_max = int(np.max(segmentation[0]))
#
#         bbox = x_min, y_min, x_max - x_min, y_max - y_min, 22
#
#     elephant_dict[elephant] = {}
#     elephant_dict[elephant]['bbox'] = bbox
#     elephant_dict[elephant]['mask'] = final_mask
#
# with open('../../data/coco_minitrain_25k/annotations/elephant_annotations.pickle', 'wb') as handle:
#     pickle.dump(elephant_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# plt.imshow(final_mask)
# plt.savefig("../../data/coco_minitrain_25k/annotations/elephant_masks/{}".format(elephant))

giraffe_dict = {}

for giraffe in tqdm(os.listdir("../../data/coco_minitrain_25k/images_pruned/giraffes_elephants/giraffe")):
    path = os.path.join(imagesRoot+'/giraffe', giraffe)

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ### from here: https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
    # create a simple mask image similar
    # to the loaded image, with the
    # shape and return type
    mask = np.zeros(image.shape[:2], np.uint8)

    # specify the background and foreground model
    # using numpy the array is constructed of 1 row
    # and 65 columns, and all array elements are 0
    # Data type for the array is np.float64 (default)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    # define the Region of Interest (ROI)
    # as the coordinates of the rectangle
    # where the values are entered as
    # (startingPoint_x, startingPoint_y, width, height)
    # these coordinates are according to the input image
    # it may vary for different images
    w, h = np.shape(image)[0], np.shape(image)[1]

    rectangle = (0, 0, w - 1, h - 1)

    # apply the grabcut algorithm with appropriate
    # values as parameters, number of iterations = 3
    # cv2.GC_INIT_WITH_RECT is used because
    # of the rectangle mode is used
    cv2.grabCut(image, mask, rectangle,
                backgroundModel, foregroundModel,
                5, cv2.GC_INIT_WITH_RECT)  # iteration number seems to be an important parameter

    # In the new mask image, pixels will
    # be marked with four flags
    # four flags denote the background / foreground
    # mask is changed, all the 0 and 2 pixels
    # are converted to the background
    # mask is changed, all the 1 and 3 pixels
    # are now the part of the foreground
    # the return type is also mentioned,
    # this gives us the final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # The final mask is multiplied with
    # the input image to give the segmented image.
    image = image * mask2[:, :, np.newaxis]

    mask_inv = cv2.bitwise_not(mask2)

    ### from here: https://stackoverflow.com/questions/72408809/how-do-i-fill-up-mask-holes-in-opencv
    # find the largest contour
    contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # draw the largest contour to fill in the holes in the mask
    final_result = np.ones(image.shape[:2])  # create a blank canvas to draw the final result
    final_result = cv2.drawContours(final_result, [largest_contour], -1, color=(0, 255, 0), thickness=cv2.FILLED)

    # show results
    final_mask = final_result.astype('uint8')
    final_mask = np.logical_not(final_mask).astype('uint8')

    # Get the bbox (xmin, ymin, width, height)
    bbox = 0, 0, 0, 0
    segmentation = np.where(final_mask == 1)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, y_min, x_max - x_min, y_max - y_min, 25

    giraffe_dict[giraffe] = {}
    giraffe_dict[giraffe]['bbox'] = bbox
    giraffe_dict[giraffe]['mask'] = final_mask

with open('../../data/coco_minitrain_25k/annotations/giraffe_annotations.pickle', 'wb') as handle:
    pickle.dump(giraffe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.imshow(final_mask)
plt.savefig("../../data/coco_minitrain_25k/annotations/giraffe_masks/{}".format(giraffe))