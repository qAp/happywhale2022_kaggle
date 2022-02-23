
import albumentations as albu


def base_tfms(image_size=64):
    return [albu.Resize(height=image_size, width=image_size)]
