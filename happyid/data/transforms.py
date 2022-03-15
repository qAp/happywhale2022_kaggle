
import cv2
import albumentations as albu


def base_tfms(image_size=64):
    return [albu.Resize(height=image_size, width=image_size)]


def aug_tfms(image_size):
    return [
        albu.Resize(height=image_size, width=image_size),
        albu.ShiftScaleRotate(shift_limit=0, scale_limit=.3, rotate_limit=30,
                              border_mode=cv2.BORDER_CONSTANT),
        albu.GaussNoise(p=.5),
        albu.Perspective(p=.2),
        albu.OneOf(
            [
                albu.CLAHE(p=.5),
                albu.RandomBrightnessContrast(p=.5),
                albu.RandomGamma(p=.2)
            ],
            p=0.4),
        albu.OneOf(
            [
                albu.Sharpen(p=.3),
                albu.Blur(blur_limit=5, p=.3),
                albu.MotionBlur(blur_limit=5, p=.3)
            ],
            p=0.5),
    ]
