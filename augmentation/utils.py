# coding: utf-8
import os
import cv2
import numpy as np
from typing import List, Tuple, Union
from matplotlib import pyplot as plt


def show_image(image: np.ndarray) -> None:
    '''image show in notebook
    '''
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title('image')
    plt.show()


def overlay_masks(
    image: np.ndarray, 
    result_image: np.ndarray, 
    result_mask: np.ndarray,
    result_image_anno: np.ndarray,
    is_mask_semantic: bool = False
) -> None:
    '''image show with masks in notebook
    '''

    f, axarr = plt.subplots(2,2, figsize=(25,15))

    axarr[0,0].imshow(image)
    axarr[0,1].imshow(result_image)
    if is_mask_semantic:
        result_mask_viz = np.zeros_like(image)
        ids = np.unique(result_mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask_viz[result_mask==id_i] = id_color
        result_mask = result_mask_viz
    
    axarr[1,0].imshow(result_mask)
    axarr[1,1].imshow(result_image_anno)

    axarr[0,0].set_aspect('auto')
    axarr[1,0].set_aspect('auto')
    axarr[0,1].set_aspect('auto')
    axarr[1,1].set_aspect('auto')

    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])

    axarr[0,1].set_xticks([])
    axarr[0,1].set_yticks([])

    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])

    axarr[1,1].set_xticks([])
    axarr[1,1].set_yticks([])

    f.subplots_adjust(wspace=0, hspace=0)

    plt.show()


def draw_bboxes(image: np.ndarray, bboxes: List[list], mask: np.ndarray=None, color: tuple=(255, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:

    result_image = image.copy()
    result_mask = np.zeros_like(result_image)

    if mask is not None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask[mask==id_i] = id_color
        cv2.addWeighted(result_mask, 0.4, result_image, 1, 0, result_image)


    for bbox in bboxes:
        if len(bbox) >= 5:
            cv2.rectangle(
                result_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])), 
                color, 
                thickness=2
            )
 
            cv2.putText(
                result_image, 
                str(bbox[4]), 
                (int(bbox[0]), int(bbox[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                thickness=2, 
                lineType=cv2.LINE_AA
            )

        else:
            bbox = [int(b) for b in bbox]

            cv2.rectangle(
                result_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]), 
                color, 
                thickness=2
            )
    
    if mask is None:
        return result_image, None
    else:
        return result_image, result_mask


def crop_bbox(image: np.ndarray, bbox: List[Union[int, float]], output_path: str=None) -> np.ndarray:
    '''crop and save a RGB image
    :param image: RGB image
    :param bbox: list of 4 voc coordinates for the bounding box
    '''
    # crop detection from image (take an additional 5 pixels around all edges)
    # cropped_img: np.ndarray = image[int(bbox[1])-5:int(bbox[3])+5, int(bbox[0])-5:int(bbox[2])+5]
    cropped_img: np.ndarray = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    if output_path:
        # save image
        cv2.imwrite(output_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    return cropped_img