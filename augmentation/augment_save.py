import cv2
import random
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Tuple
from .copy_paste import CopyPaste, CopyPasteAugmentation

# ------------------GENERAL CONFIG----------------------
# foreground objects
object_folder = Path(r'C:\Users\RN767KA\Projects\Document Classification\data\foreground\driver licence')
# background images
background_folder = Path(r'C:\Users\RN767KA\Projects\Document Classification\data\background')
# output
n_output = 10
output_dir = r'C:\Users\RN767KA\Projects\Document Classification\data\augmented_data'

# the key of the map must match the foreground folder name
# for yolo, the pairs must match the .names file
class_mapping = {
    'driver licence': 0,
    'passport': 1
}

# ------------------TRANSFORMATION CONFIG----------------------
# define foreground augmentations
elastic_transform = A.Compose([
    A.RandomRotate90(
        always_apply=False, 
        p=0.3
    ),
    A.Rotate(
        always_apply=False, 
        p=0.9, 
        limit=(-15, 15), 
        interpolation=0, 
        border_mode=0, 
        value=(0, 0, 0), 
        mask_value=None
    ),
    A.ElasticTransform(
        always_apply=False, 
        p=0.5, 
        alpha=1.0, 
        sigma=50.0, 
        alpha_affine=10.0, 
        interpolation=0, 
        border_mode=0, 
        value=(0, 0, 0), 
        mask_value=None, 
        approximate=False
    ),
    A.OpticalDistortion(
        always_apply=False, 
        p=0.3, 
        distort_limit=(-0.3, 0.3), 
        shift_limit=(-0.05, 0.05), 
        interpolation=0, 
        border_mode=0, 
        value=(0, 0, 0), 
        mask_value=None
    )
])

# define background augmentation
pixel_transform = A.ReplayCompose(
    [
        A.RandomBrightnessContrast(
            always_apply=False, 
            p=1.0, 
            brightness_limit=(-0.2, 0.2), 
            contrast_limit=(-0.2, 0.2), 
            brightness_by_max=True
        ),
        A.GaussNoise(always_apply=False, p=0.6, var_limit=(10.0, 300)),
        A.MotionBlur(always_apply=False, p=0.1, blur_limit=(3, 7)),
        A.ToGray(always_apply=False, p=0.05),
        A.ImageCompression(
            always_apply=False, 
            p=0.8, 
            quality_lower=30, 
            quality_upper=80, 
            compression_type=0
        )
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc', 
        # min_area=1024, 
        # min_visibility=0.5, 
        label_fields=['class_labels']
    )
)
# ------------------END of CONFIGS----------------------

def voc_to_yolo(bbox: Tuple[float, float, float, float], size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    '''
    :param bbox: a tuple of pascal voc bounding box (x_min, y_min, x_max, y_max)
    :param size: a tuple of 2D dimension (width, height)

    return: a tuple of yolo normalised bounding box NORMALISED(x_center, y_center, width, height)
    '''
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (bbox[0] + bbox[2]) / 2.0 - 1
    y = (bbox[1] + bbox[3]) / 2.0 - 1
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def augment(n: int, output_dir: str, format: str='yolo') -> None:
    '''
    output n augmented images with predefined formats for model training

    :param n: total number of images to be generated
    '''
    # load set of foregrounds
    class_name = object_folder.name
    foreground_images = list(object_folder.glob('*.png')) + list(object_folder.glob('*.jpg'))
    # load set of backgrounds
    background_images = list(background_folder.rglob('*.jpg'))
    # prepare output folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    # total number of random choices
    while count < n:
        # randomly choose a background
        bg_image_fp = random.choice(background_images)
        bg_image: np.ndarray = cv2.imread(str(bg_image_fp))
        height, width, channels = bg_image.shape

        # define foreground/background interaction
        # TODO:
        # 1. The initial foreground transformation will crop the edges
        # 2. Current copy paste ignore the min visibility
        # 3. cv2.error: OpenCV(4.5.4) D:\a\opencv-python\opencv-python\opencv\modules\core\src\arithm.cpp:230: error: (-215:Assertion faile
        overlay_background = CopyPasteAugmentation(
            source_images=foreground_images,
            albu_transforms=elastic_transform,
            bev_transform=None, 
            n_objects_range=[1, 1], # object count
            h_range=[0.2 * height, 0.8 * height], # object height -> size
            x_range=[0.2 * width, 0.8 * width], # x center
            y_range=[0.3 * height, 1 * height] # y bottom
        )

        try:
            # overlay foreground onto background, omit mask output
            result_image, result_coords, _, _ = overlay_background(bg_image)
            transformed = pixel_transform(
                image=result_image, 
                bboxes=result_coords, 
                class_labels=[class_name] * len(result_coords), # use folder name as class label
                # class_categories=class_categories
            )
        except Exception as e:
            print(e)
            continue

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # save the images
        transformed_image_fp = Path(output_dir) / f'{class_name}_{count}.jpg'
        cv2.imwrite(str(transformed_image_fp), transformed_image)

        if format == 'yolo':
            transformed_bboxes_fp = transformed_image_fp.with_suffix('.txt')
            yolo_bboxes = [voc_to_yolo(bbox, (width, height)) for bbox in transformed_bboxes]
            # overwrite the text file
            with transformed_bboxes_fp.open('w') as f:
                for j, yolo_box in enumerate(yolo_bboxes):
                    details = (
                        str(class_mapping[transformed_class_labels[j]]) # class index
                        + ' '
                        + ' '.join(str(coord) for coord in yolo_box)
                        + '\n'
                    )
                    f.write(details)

        count += 1

if __name__ == '__main__':
    augment(n_output, output_dir)
    print('Done')