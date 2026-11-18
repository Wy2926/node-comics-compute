"""Hayai paragraph bounds and the native 48px color model's input limit."""
OCR_MAX_WIDTH = 3200


def valid_crop(image):
    return (image is not None and image.size > 0
            and image.shape[1] * 48 <= image.shape[0] * OCR_MAX_WIDTH)


def paragraph_box(box, width, height):
    left, top, right, bottom = map(int, box)
    return max(0, left), max(0, top), min(width, right), min(height, bottom)
