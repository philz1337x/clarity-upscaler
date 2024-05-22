import numpy as np
import cv2
import mediapipe as mp
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw

def detect_and_crop_hand_from_binary(binary_image_data):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Convert binary data to a NumPy array
    np_arr = np.frombuffer(binary_image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image from binary data.")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list, y_list = [], []
            h, w, _ = img.shape
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_list.append(x)
                y_list.append(y)
            
            # Calculate the bounding box
            min_x, min_y = min(x_list), min(y_list)
            max_x, max_y = max(x_list), max(y_list)
            
            # Increase the size of the bounding box by 50%
            width, height = max_x - min_x, max_y - min_y
            min_x = max(0, min_x - int(width * 0.25))
            min_y = max(0, min_y - int(height * 0.25))
            max_x = min(w, max_x + int(width * 0.25))
            max_y = min(h, max_y + int(height * 0.25))

            # Ensure the bounding box is at least 512x512
            if (max_x - min_x) < 512:
                center_x = (min_x + max_x) // 2
                min_x = max(0, center_x - 256)
                max_x = min(w, center_x + 256)
            
            if (max_y - min_y) < 512:
                center_y = (min_y + max_y) // 2
                min_y = max(0, center_y - 256)
                max_y = min(h, center_y + 256)

            if max_x - min_x < 512:
                if min_x == 0:
                    max_x = min(w, 512)
                elif max_x == w:
                    min_x = max(0, w - 512)
                else:
                    max_x = min(w, min_x + 512)
            
            if max_y - min_y < 512:
                if min_y == 0:
                    max_y = min(h, 512)
                elif max_y == h:
                    min_y = max(0, h - 512)
                else:
                    max_y = min(h, min_y + 512)

            cropped_img = img[min_y:max_y, min_x:max_x]

            return cropped_img, (min_x, min_y, max_x, max_y)
    
    return None, None

def create_mask(image):
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Invalid input. Please provide an image file path or a PIL Image object.")

    np.array(img)

    mask = Image.new("RGB", img.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    width, height = img.size

    rect_width = int(width * 0.8)
    rect_height = int(height * 0.8)
    start_x = int((width - rect_width) / 2)
    start_y = int((height - rect_height) / 2)
    end_x = start_x + rect_width
    end_y = start_y + rect_height

    draw.rectangle([start_x, start_y, end_x, end_y], fill=(255, 255, 255))

    border_size = int(width * 0.1)
    draw.rectangle([0, 0, width, border_size], fill=(0, 0, 0))  # Oben
    draw.rectangle([0, height - border_size, width, height], fill=(0, 0, 0))  # Unten
    draw.rectangle([0, 0, border_size, height], fill=(0, 0, 0))  # Links
    draw.rectangle([width - border_size, 0, width, height], fill=(0, 0, 0))  # Rechts

    mask = mask.filter(ImageFilter.GaussianBlur(15))

    return mask

def combine_hands(orig_hands, img_hands_post, mask):
    src1 = np.array(orig_hands)
    src2 = np.array(img_hands_post)
    mask1 = np.array(mask)
    mask1 = mask1 / 255
    dst = src2 * mask1 + src1 * (1 - mask1)
    smooth_face = Image.fromarray(dst.astype(np.uint8))
    return  smooth_face

def insert_image(reference_image, small_image, top, left):
    small_width, small_height = small_image.size
    box = (left, top, left + small_width, top + small_height)
    reference_image.paste(small_image, box)
    return reference_image

def insert_cropped_hand_into_image(original_img_data, cropped_img_object, coords, cropped_hand_img_pil, scale_factor=2):
    original_img = Image.open(BytesIO(original_img_data)).convert("RGBA")

    cropped_img = cropped_img_object
    cropped_img = cropped_img.resize((cropped_img.width // scale_factor, cropped_img.height // scale_factor))
    
    mask = create_mask(cropped_img)

    img_smooth_hands = combine_hands(cropped_hand_img_pil, cropped_img, mask)

    new_image = insert_image(original_img, img_smooth_hands, coords[1], coords[0])
    new_image_rgb = new_image.convert("RGB")

    return new_image_rgb