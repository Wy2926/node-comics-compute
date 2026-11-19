"""Hayai paragraph crop bounds."""


def paragraph_box(box, width, height):
    left, top, right, bottom = map(int, box)
    return max(0, left), max(0, top), min(width, right), min(height, bottom)
