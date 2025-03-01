from importlib import import_module
from .ddim_depth_estimate_res_swin_add import DDIMDepthEstimate_Swin_ADD
from .ddim_depth_estimate_res_swin_addHAHI import DDIMDepthEstimate_Swin_ADDHAHI
from .ddim_depth_estimate_res_swin_addHAHI_vis import DDIMDepthEstimate_Swin_ADDHAHIVis
from .ddim_depth_estimate_res_mpvit_HAHI import DDIMDepthEstimate_MPVIT_ADDHAHI
from .ddim_depth_estimate_res import DDIMDepthEstimate_Res
from .ddim_depth_estimate_tode import DDIMDepthEstimate_TODE
from .ddim_depth_estimate_res_vis import DDIMDepthEstimate_ResVis
from .vae_tode import vae_tode
from .tode_head import Tode_Head
from .tode_head_swin import Tode_Head_Swin
from .double_tode import Double_Tode
from .tode_rgbdn_head import Tode_RGBDN_Head
from .unconditional_unet_head import Unconditional_Unet_head
def get(args):
    model_name = args.head_name
    module_name = 'model.' + model_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)
