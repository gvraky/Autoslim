import torch #line:1
import torch .nn as nn #line:2
from copy import deepcopy #line:3
__all__ =['mask_weight','mask_bias']#line:5
def _O0O000O000000OO0O (module ,input ):#line:7
    if hasattr (module ,'weight_mask'):#line:8
        module .weight .data *=module .weight_mask #line:9
def _OO0OOOO0OOOOO00OO (module ,input ):#line:11
    if module .bias is not None and hasattr (module ,'bias_mask'):#line:12
        module .bias .data *=module .bias_mask #line:13
def mask_weight (layer ,mask ,inplace =True ):#line:15
    ""#line:21
    if not inplace :#line:22
        layer =deepcopy (layer )#line:23
    if mask .shape !=layer .weight .shape :#line:24
        return layer #line:25
    mask =torch .tensor (mask ,dtype =layer .weight .dtype ,device =layer .weight .device ,requires_grad =False )#line:26
    if hasattr (layer ,'weight_mask'):#line:27
        mask =mask +layer .weight_mask #line:28
        mask [mask >0 ]=1 #line:29
        layer .weight_mask =mask #line:30
    else :#line:31
        layer .register_buffer ('weight_mask',mask )#line:32
    layer .register_forward_pre_hook (_O0O000O000000OO0O )#line:34
    return layer #line:35
def mask_bias (layer ,mask ,inplace =True ):#line:37
    ""#line:43
    if not inplace :#line:44
        layer =deepcopy (layer )#line:45
    if layer .bias is None or mask .shape !=layer .bias .shape :#line:46
        return layer #line:47
    mask =torch .tensor (mask ,dtype =layer .weight .dtype ,device =layer .weight .device ,requires_grad =False )#line:49
    if hasattr (layer ,'bias_mask'):#line:50
        mask =mask +layer .bias_mask #line:51
        mask [mask >0 ]=1 #line:52
        layer .bias_mask =mask #line:53
    else :#line:54
        layer .register_buffer ('bias_mask',mask )#line:55
    layer .register_forward_pre_hook (_OO0OOOO0OOOOO00OO )#line:56
    return layer #line:57
