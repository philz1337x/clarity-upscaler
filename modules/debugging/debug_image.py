import os
import platform
import shutil
from cog import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from modules.tiling.img_utils import expand_canvas_tiling

def save_output_img(pil_img, filename, info_text="", add_debug_info=True):
    filepath = Path(filename) ## NOTE this is a cog.Path!!

    ## make a copy of the image.
    img_copy = pil_img.copy()

    if info_text:
        img_copy = burn_text_into_image(img_copy, info_text)

    if add_debug_info:
        img_copy = draw_debug_info_into_img(img_copy)

    img_copy.save(filepath)

    return filepath

def burn_text_into_image(pil_img, text_str, pos_x=50, pos_y=50, font_size=22, font_name='arial.ttf'):
    """
    make font size resolution independent, based on image of 1024x1024
    """
    width, height = pil_img.size

    ## lets just base it on the width of the image for now.
    font_size = calc_relative(width, font_size)

    draw = ImageDraw.Draw(pil_img, 'RGBA')

    ## replace the font_name in the case we're on linux as we have to
    ## pass a full path to the font file.
    if platform.system()=='Linux':
        font_name = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(font_name, size=font_size)
    ## now lets make these positions relative.
    text_pos_x = calc_relative(width, pos_x)
    text_pos_y = calc_relative(width, pos_y)
    text_position = (text_pos_x, text_pos_y)
    text_color = 'rgb(255, 255, 255)'

    ## draw a black backdrop to make text more visible
    expand_size = calc_relative(width, 10)
    draw_text_backdrop(text_str, draw, font, text_position, expand=expand_size)

    ## now draw the text onto the image
    draw.text(text_position, text_str, fill=text_color, font=font)

    return pil_img

def draw_text_backdrop(text_str, draw_obj, font_obj, pos, expand=10):
    text_pos_x, text_pos_y = pos
    # Determine the size of the text
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = font_obj.getbbox(text_str)
    bbox_x1 += text_pos_x - expand
    bbox_y1 += text_pos_y - expand
    bbox_x2 += text_pos_x + expand
    bbox_y2 += text_pos_y + expand
    text_bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
    # Draw the semi-transparent rectangle
    draw_obj.rectangle(text_bbox, fill=(0, 0, 0, 128))  # 128 out of 255 for 50% opacity


def draw_border(img, color='red', offset=1, thickness=1, darken=True):
    draw = ImageDraw.Draw(img, 'RGBA')
    width, height = img.size

    ## if darken is turned on, we draw a dark rectangle in the middle of the border.
    if darken:
        draw.rectangle([(offset, offset), (width - offset, height - offset)],
                       fill=(0, 0, 0, 128))  # 128 out of 255 for 50% opacity

    ## now draw the border and return the image
    # Draw multiple rectangles for thicker borders
    for i in range(thickness):
        draw.rectangle([(offset - i, offset - i), (width - offset - i, height - offset - i)], outline=color)

    return img


def calc_relative(width, abs_value):
    """
    function can be used to calculate any kind of relative sizes,
    pixel offsets or font sizes
    uses 1024 as the guidance size, and will increase or decrease
    the relative size according to the provided width of the image.
    """
    scale_ratio = float(width) / 1024.0
    relative_value= int(abs_value * scale_ratio)
    return relative_value

def debug_tiling_image(img):
    debug_img = img.copy()
    border_offset = calc_relative(debug_img.width, 100)
    thickness = calc_relative(debug_img.width, 5)
    draw_border(debug_img, color='black', offset=border_offset, thickness=thickness)
    ## first lets make these images tile..
    border_size = calc_relative(debug_img.width, 256)
    tiling_img = expand_canvas_tiling(debug_img, darken=False)
    return tiling_img

def draw_debug_info_into_img(img):
    width, height = img.size
    img = burn_text_into_image(img, f'res: {width}x{height}px', pos_y=100)
    return img

def move_files_to_timestamped_subdirectory(source_directory):
    # Get the current date and time formatted as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create a new directory name with this timestamp
    destination_directory = os.path.join(source_directory, timestamp)

    # Create the directory if it does not exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # List all files in the source directory
    files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

    # Move each file to the new directory
    for file in files:
        shutil.move(os.path.join(source_directory, file), os.path.join(destination_directory, file))

