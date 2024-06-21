import base64
import numpy as np
from io import BytesIO
from PIL import ImageOps, ImageEnhance, Image, ImageDraw, ImageFilter

def add_border(img, border_size):
    img = img.copy()
    # Define the border size: (left, top, right, bottom)
    border = (border_size, border_size, border_size, border_size)

    # Create a new image with the border
    # The expand argument in the border method adds the border on the outside of the image
    bordered_img = ImageOps.expand(img, border=border, fill='black')

    return bordered_img

def crop_corner(img, crop_size, corner='bottom_right', darken=False):
    # Open the original image
    img = img.copy()
    width, height = img.size

    # Initialize coordinates
    left, top, right, bottom = 0, 0, 0, 0

    if corner == 'bottom_right':
        left = width - crop_size
        top = height - crop_size
        right = width
        bottom = height
    elif corner == 'bottom_left':
        left = 0
        top = height - crop_size
        right = crop_size
        bottom = height
    elif corner == 'top_right':
        left = width - crop_size
        top = 0
        right = width
        bottom = crop_size
    elif corner == 'top_left':
        left = 0
        top = 0
        right = crop_size
        bottom = crop_size
    else:
        raise ValueError("Invalid corner. Choose from 'bottom_right', 'bottom_left', 'top_right', or 'top_left'.")

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    if darken:
        cropped_img = darken_image(cropped_img, 0.8)

    return cropped_img

def crop_side(img, crop_size, side, darken=False):
    # Open the original image
    img = img.copy()
    width, height = img.size

    # Initialize coordinates
    left, top, right, bottom = 0, 0, width, height

    if side == 'left':
        right = crop_size
    elif side == 'right':
        left = width - crop_size
    elif side == 'bottom':
        top = height - crop_size
    elif side == 'top':
        bottom = crop_size
    else:
        raise ValueError("Invalid side. Choose from 'left', 'right', 'bottom', or 'top'.")

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    if darken:
        cropped_img = darken_image(cropped_img, 0.4)

    return cropped_img

def darken_image(img, factor=0.5):
    # Create a brightness enhancer
    enhancer = ImageEnhance.Brightness(img)

    # Apply the enhancer with the given factor (0.5 to darken the image by 50%)
    darkened_img = enhancer.enhance(factor)

    return darkened_img

def calculate_border_size(img, border_pct=5, mode='scaleup'):
    """
    ** function assumes square image **
    mode can be either 'scaleup' or 'cropback'
    - scale up calculates border size based on percentage of current image
    - crop back calculates the border size based on a pre-expanded image,
      thus will give us the border size we want to crop back from.
    """
    if isinstance(img, bytes):
        img = convert_binary_img_to_pil(img)

    width, height = img.size
    border_size = 0 ## init to 0
    if mode=='scaleup':
        border_size = int(width * (border_pct / 100.0))
    elif mode=='cropback':
        border_size = int(width * (border_pct / (100.0 + border_pct + border_pct)))

    return border_size

def expand_canvas_tiling(img, div=8, darken=True):
    width, height = img.size
    org_size = width
    border_size = int(org_size / div)

    ## first add a black border around the image to expand the canvas
    expanded_img = add_border(img, border_size)

    ## now crop the bottom right, which will become the top left
    tl_img = crop_corner(img, border_size, corner='bottom_right', darken=darken)
    tl_x = 0
    tl_y = 0
    ## and the top left which will be place on the bottom right
    br_img = crop_corner(img, border_size, corner='top_left', darken=darken)
    br_x = org_size + border_size
    br_y = org_size + border_size
    ## and the bottom left which will be placed to the top right
    tr_img = crop_corner(img, border_size, corner='bottom_left', darken=darken)
    tr_x = org_size + border_size
    tr_y = 0
    ## and the top right which will be placed to the bottom left
    bl_img = crop_corner(img, border_size, corner='top_right', darken=darken)
    bl_x = 0
    bl_y = org_size + border_size

    ## now crop the sides.
    left_img = crop_side(img, border_size, 'left', darken=darken)
    left_x = org_size + border_size
    left_y = border_size
    ## and the right
    right_img = crop_side(img, border_size, 'right', darken=darken)
    right_x = 0
    right_y = border_size
    ## and the top
    top_img = crop_side(img, border_size, 'top', darken=darken)
    top_x = border_size
    top_y = org_size + border_size
    ## and the bottom
    bottom_img = crop_side(img, border_size, 'bottom', darken=darken)
    bottom_x = border_size
    bottom_y = 0

    expanded_img.paste(tl_img, (tl_x, tl_y))
    expanded_img.paste(br_img, (br_x, br_y))
    expanded_img.paste(tr_img, (tr_x, tr_y))
    expanded_img.paste(bl_img, (bl_x, bl_y))

    expanded_img.paste(left_img, (left_x, left_y))
    expanded_img.paste(right_img, (right_x, right_y))
    expanded_img.paste(top_img, (top_x, top_y))
    expanded_img.paste(bottom_img, (bottom_x, bottom_y))

    return expanded_img

def convert_binary_img_to_pil(binary_img):
    pil_image = Image.open(BytesIO(binary_img))
    return pil_image

def convert_pil_img_to_binary(pil_img):
    imgbuffer = BytesIO()
    pil_img.save(imgbuffer, format='PNG')
    binary_img = imgbuffer.getvalue()
    return binary_img

def convert_pil_img_to_base64(pil_img, force_rgb=True):
    if pil_img.mode == 'RGBA' and force_rgb:
        pil_img = pil_img.convert('RGB')
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_str

def convert_base64_img_to_pil(base64_img):
    binary_img = BytesIO(base64.b64decode(base64_img))
    pil_img = Image.open(binary_img)
    return pil_img

def shift_image(img, x, y):
    """shifts the pixels, wrapping around itself"""
    np_img = np.array(img) ### convert to a numpy array, as we receive a PIL image.
    np_img = np.roll(np_img, shift=x, axis=1)
    np_img = np.roll(np_img, shift=y, axis=0)
    return_img = Image.fromarray(np_img) ## convert numpy array back to PIL image.
    return return_img

def draw_center_cross_image(img=None, thickness_mult=1.0, blur_mult=1.0, offset_x=0, offset_y=0,
                            x_start=0, x_end=0, y_start=0, y_end=0, boost=True):
    if img:
        width, height = img.size
        img = Image.new('RGB', (width, height), (0, 0, 0))
    else:
        img = Image.new('RGB', (1280, 720), (0, 0, 0))

    draw = ImageDraw.Draw(img, 'RGB')
    width, height = img.size

    ## now calculate the line thickness based on the resolution of the image
    thickness = int((width / 25) * thickness_mult)
    ## and calculate the blur_radius based on the thickness of the line.
    blur_radius = thickness * 0.4 * blur_mult
    print(f'blur radius is {blur_radius}')

    ## now prepare some other values we will use
    half_thickness = int(thickness / 2.0)
    half_width = int(width / 2.0)
    half_height = int(height / 2.0)
    offset = thickness / 2 ## the offset of the line from the edge of the image.

    ## first lets draw it vertically.
    for i in range(thickness):
        x1 = half_width - half_thickness + i + offset_x
        y1 = y_start or (0 + offset)
        x2 = half_width - half_thickness + i + offset_x
        y2 = y_end or (height - offset)
        draw.rectangle([(x1, y1), (x2, y2)], outline='white')

    ## and now lets draw the horizontal cross bar.
    for i in range(thickness):
        x1 = x_start or (0 + offset)
        y1 = half_height - half_thickness + i + offset_y
        x2 = x_end or (width - offset)
        y2 = half_height - half_thickness + i + offset_y
        draw.rectangle([(x1, y1), (x2, y2)], outline='white')

    img_blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img_blur.show()

    if boost:
        ## finally lets lift up the whites a little that might have been affected by the blur.
        lifted_img = ImageOps.autocontrast(img_blur, cutoff=2)
        return lifted_img

    return img_blur