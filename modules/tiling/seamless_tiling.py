from PIL import ImageOps, ImageEnhance, Image
from modules.tiling.img_utils import convert_binary_img_to_pil, \
                                    convert_pil_img_to_binary, \
                                    expand_canvas_tiling

def preprocess_expand_canvas_for_tile_image(img, border_size=128, darken=False, force_original_res=True):
    ## first lets check the data type of the image, it could be binary or a PIL.Image...
    if isinstance(img, bytes):
        input_type = 'binary'
        img = convert_binary_img_to_pil(img)
    elif isinstance(img, Image.Image):
        ### then it's a PIL.Image.
        input_type = 'pil'

    # Specify the path to your original image and the output path
    width, height = img.size
    org_size = width

    expanded_img = expand_canvas_tiling(img, border_size=border_size, darken=darken)

    if force_original_res:
        expanded_img = expanded_img.resize((width, height), Image.Resampling.LANCZOS)
        pass

    ## finally if the input_type was binary, we need to convert the pil img back to binary
    if input_type=='binary':
        expanded_img = convert_pil_img_to_binary(expanded_img)

    return expanded_img

def postprocess_crop_canvas_back(img, border_size=128):
    ## check if the incoming image is binary or pil
    if isinstance(img, bytes):
        input_type = 'binary'
        img = convert_binary_img_to_pil(img)
    elif isinstance(img, Image.Image):
        input_type = 'pil'
    ## get the image size
    width, height = img.size
    ## now crop back the image
    cropped_img = img.crop((border_size, border_size, width - border_size, height - border_size))

    ## if the input was binary, make sure to convert it back to binary
    if input_type=='binary':
        cropped_img = convert_pil_img_to_binary(cropped_img)

    return cropped_img



