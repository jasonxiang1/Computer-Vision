import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

#delete library later
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    #denoising/blurring step removed
    #taking too long

    #convert image to grayscale
    image_gray = skimage.color.rgb2gray(image)

    #encompassed in black
    #image_gray = np.invert(np.uint8(image_gray*255))

    # apply threshold
    thresh = skimage.filters.threshold_otsu(image)
    print(thresh)
    #maybe use opening
    #change size
    bw = skimage.morphology.closing(image_gray < thresh, skimage.morphology.square(3))

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)

    # label image regions
    label_image = skimage.measure.label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    # image_label_overlay = skimage.color.label2rgb(label_image, image=image, bg_label=0)
    #
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    #declare array to store bounding box parameters
    bboxes_store = np.zeros((1,4))

    #store top left and bottom right images of regions
    for region in skimage.measure.regionprops(label_image):
        if region.area >= 325:
            tl_r, tl_c, br_r, br_c = region.bbox
            bboxes_store = np.append(bboxes_store,np.array([tl_r, tl_c, br_r, br_c])[np.newaxis,:],axis=0)

    bboxes =bboxes_store[1:,:]

    #to display bounding boxes
    # for region in skimage.measure.regionprops(label_image):
    #     # take regions with large enough areas
    #     if region.area >= 100:
    #         # draw rectangle around segmented coins
    #         minr, minc, maxr, maxc = region.bbox
    #         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                   fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)
    #
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()

    return bboxes, bw