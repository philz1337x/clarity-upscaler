import sys
#sys.path.extend(['/stable-diffusion-webui'])

#from modules import timer
from modules import launch_utils

with launch_utils.startup_timer.subcategory("prepare environment"):
    launch_utils.prepare_environment()