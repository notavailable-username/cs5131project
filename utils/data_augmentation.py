import cv2
import numpy as np
import random

def augment_image(image, config):
    # Apply random rotation
    angle = random.uniform(-config.get("rotation_range", 15), config.get("rotation_range", 15))
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Apply horizontal flip if enabled
    if config.get("horizontal_flip", True) and random.random() < 0.5:
        rotated = cv2.flip(rotated, 1)
    
    # Apply brightness adjustment
    brightness_limit = config.get("brightness_limit", 0.2)
    factor = 1.0 + random.uniform(-brightness_limit, brightness_limit)
    augmented = cv2.convertScaleAbs(rotated, alpha=factor, beta=0)
    
    return augmented

if __name__ == "__main__":
    import cv2
    image = cv2.imread("dummy.jpg")
    config = {"rotation_range": 15, "horizontal_flip": True, "brightness_limit": 0.2}
    aug = augment_image(image, config)
    cv2.imshow("Augmented", aug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
