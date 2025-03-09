import cv2
import numpy as np
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

def augment_image(image, config):
    """
    Apply augmentations to an image.
    
    Args:
        image: Input image as numpy array
        config: Dictionary with augmentation parameters
        
    Returns:
        Augmented image
    """
    if image is None or image.size == 0:
        logger.error("Invalid image provided for augmentation")
        raise ValueError("Invalid image provided for augmentation")
        
    try:
        # Create a copy to avoid modifying the original
        augmented = image.copy()
        
        # Apply random rotation
        if "rotation_range" in config and config["rotation_range"] > 0:
            angle = random.uniform(-config.get("rotation_range"), config.get("rotation_range"))
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            logger.debug(f"Applied rotation of {angle:.2f} degrees")
        
        # Apply horizontal flip if enabled
        if config.get("horizontal_flip", False) and random.random() < 0.5:
            augmented = cv2.flip(augmented, 1)
            logger.debug("Applied horizontal flip")
        
        # Apply brightness adjustment
        if "brightness_limit" in config and config["brightness_limit"] > 0:
            brightness_limit = config.get("brightness_limit")
            factor = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)
            logger.debug(f"Applied brightness adjustment with factor {factor:.2f}")
        
        return augmented
        
    except Exception as e:
        logger.error(f"Error during image augmentation: {e}")
        raise

if __name__ == "__main__":
    import argparse
    import os
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test Image Augmentation")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output directory for augmented images')
    parser.add_argument('--count', type=int, default=5, help='Number of augmented versions to create')
    parser.add_argument('--rotation', type=float, default=15, help='Max rotation in degrees')
    parser.add_argument('--brightness', type=float, default=0.2, help='Brightness adjustment limit')
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flipping')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        exit(1)
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
    # Load input image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        exit(1)
    
    # Configure augmentation
    config = {
        "rotation_range": args.rotation,
        "horizontal_flip": not args.no_flip,
        "brightness_limit": args.brightness
    }
    
    print(f"Generating {args.count} augmented versions with:")
    print(f"  - Rotation range: ±{args.rotation} degrees")
    print(f"  - Brightness adjustment: ±{args.brightness * 100}%")
    print(f"  - Horizontal flip: {'disabled' if args.no_flip else 'enabled'}")
    
    # Generate augmented images
    for i in range(args.count):
        try:
            augmented = augment_image(image, config)
            
            if args.output:
                output_path = os.path.join(args.output, f"augmented_{i:03d}.jpg")
                cv2.imwrite(output_path, augmented)
                print(f"Saved: {output_path}")
            else:
                # Display if no output directory
                cv2.imshow(f"Augmented {i+1}", augmented)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error generating augmentation {i+1}: {e}")
    
    print("Augmentation complete.")
