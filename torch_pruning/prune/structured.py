import torch #line:1
import torch .nn as nn #line:2
from copy import deepcopy #line:3
from functools import reduce #line:4
from operator import mul #line:5
__all__ =['prune_conv','prune_related_conv','prune_linear','prune_related_linear','prune_batchnorm','prune_prelu','prune_group_conv']#line:7
def prune_group_conv (layer :nn .modules .conv ._ConvNd ,idxs :list ,inplace :bool =True ,dry_run :bool =False ):#line:9
    ""#line:15
    if layer .groups >1 :#line:16
         assert layer .groups ==layer .in_channels and layer .groups ==layer .out_channels ,"only group conv with in_channel==groups==out_channels is supported"#line:17
    idxs =list (set (idxs ))#line:19
    OO0OOOO0OOOOO0OO0 =len (idxs )*reduce (mul ,layer .weight .shape [1 :])+(len (idxs )if layer .bias is not None else 0 )#line:20
    if dry_run :#line:21
        return layer ,OO0OOOO0OOOOO0OO0 #line:22
    if not inplace :#line:23
        layer =deepcopy (layer )#line:24
    OO00OOO0O0O0OOOOO =[OOO0OOOOO0O00OOO0 for OOO0OOOOO0O00OOO0 in range (layer .out_channels )if OOO0OOOOO0O00OOO0 not in idxs ]#line:25
    layer .out_channels =layer .out_channels -len (idxs )#line:26
    layer .in_channels =layer .in_channels -len (idxs )#line:27
    layer .groups =layer .groups -len (idxs )#line:28
    layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[OO00OOO0O0O0OOOOO ])#line:29
    if layer .bias is not None :#line:30
        layer .bias =torch .nn .Parameter (layer .bias .data .clone ()[OO00OOO0O0O0OOOOO ])#line:31
    return layer ,OO0OOOO0OOOOO0OO0 #line:32
def prune_conv (layer :nn .modules .conv ._ConvNd ,idxs :list ,inplace :bool =True ,dry_run :bool =False ):#line:34
    ""#line:40
    idxs =list (set (idxs ))#line:41
    OO000O0000OO0OO00 =len (idxs )*reduce (mul ,layer .weight .shape [1 :])+(len (idxs )if layer .bias is not None else 0 )#line:42
    if dry_run :#line:43
        return layer ,OO000O0000OO0OO00 #line:44
    if not inplace :#line:46
        layer =deepcopy (layer )#line:47
    OO00O00OOOOO0O00O =[O0000OOOO00OOO0O0 for O0000OOOO00OOO0O0 in range (layer .out_channels )if O0000OOOO00OOO0O0 not in idxs ]#line:49
    layer .out_channels =layer .out_channels -len (idxs )#line:50
    if isinstance (layer ,(nn .ConvTranspose2d ,nn .ConvTranspose3d )):#line:51
        layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[:,OO00O00OOOOO0O00O ])#line:52
    else :#line:53
        layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[OO00O00OOOOO0O00O ])#line:54
    if layer .bias is not None :#line:55
        layer .bias =torch .nn .Parameter (layer .bias .data .clone ()[OO00O00OOOOO0O00O ])#line:56
    return layer ,OO000O0000OO0OO00 #line:57
def prune_related_conv (layer :nn .modules .conv ._ConvNd ,idxs :list ,inplace :bool =True ,dry_run :bool =False ):#line:59
    ""#line:65
    idxs =list (set (idxs ))#line:66
    O0O0O0O000OO000O0 =len (idxs )*layer .weight .shape [0 ]*reduce (mul ,layer .weight .shape [2 :])#line:67
    if dry_run :#line:68
        return layer ,O0O0O0O000OO000O0 #line:69
    if not inplace :#line:70
        layer =deepcopy (layer )#line:71
    O0O00O0OO0000O00O =[OO0O0O00O00O0OO0O for OO0O0O00O00O0OO0O in range (layer .in_channels )if OO0O0O00O00O0OO0O not in idxs ]#line:74
    layer .in_channels =layer .in_channels -len (idxs )#line:76
    if isinstance (layer ,(nn .ConvTranspose2d ,nn .ConvTranspose3d )):#line:78
        layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[O0O00O0OO0000O00O ,:])#line:79
    else :#line:80
        layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[:,O0O00O0OO0000O00O ])#line:81
    return layer ,O0O0O0O000OO000O0 #line:83
def prune_linear (layer :nn .modules .linear .Linear ,idxs :list ,inplace :list =True ,dry_run :list =False ):#line:85
    ""#line:91
    OOO00OO00O000OOO0 =len (idxs )*layer .weight .shape [1 ]+(len (idxs )if layer .bias is not None else 0 )#line:92
    if dry_run :#line:93
        return layer ,OOO00OO00O000OOO0 #line:94
    if not inplace :#line:96
        layer =deepcopy (layer )#line:97
    O0OOOOOOO000000OO =[OOO0OO000OO0000O0 for OOO0OO000OO0000O0 in range (layer .out_features )if OOO0OO000OO0000O0 not in idxs ]#line:98
    layer .out_features =layer .out_features -len (idxs )#line:99
    layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[O0OOOOOOO000000OO ])#line:100
    if layer .bias is not None :#line:101
        layer .bias =torch .nn .Parameter (layer .bias .data .clone ()[O0OOOOOOO000000OO ])#line:102
    return layer ,OOO00OO00O000OOO0 #line:103
def prune_related_linear (layer :nn .modules .linear .Linear ,idxs :list ,inplace :list =True ,dry_run :list =False ):#line:105
    ""#line:111
    O00O00O00O0000O0O =len (idxs )*layer .weight .shape [0 ]#line:112
    if dry_run :#line:113
        return layer ,O00O00O00O0000O0O #line:114
    if not inplace :#line:116
        layer =deepcopy (layer )#line:117
    OO0OOOO0O0O0000OO =[O00OO0O000O0O0O0O for O00OO0O000O0O0O0O in range (layer .in_features )if O00OO0O000O0O0O0O not in idxs ]#line:118
    layer .in_features =layer .in_features -len (idxs )#line:119
    layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[:,OO0OOOO0O0O0000OO ])#line:120
    return layer ,O00O00O00O0000O0O #line:121
def prune_batchnorm (layer :nn .modules .batchnorm ._BatchNorm ,idxs :list ,inplace :bool =True ,dry_run :bool =False ):#line:123
    ""#line:129
    OO0000O0O0OOOOOO0 =len (idxs )*(2 if layer .affine else 1 )#line:131
    if dry_run :#line:132
        return layer ,OO0000O0O0OOOOOO0 #line:133
    if not inplace :#line:135
        layer =deepcopy (layer )#line:136
    OO00000O0000O0O0O =[O000O00000O00O000 for O000O00000O00O000 in range (layer .num_features )if O000O00000O00O000 not in idxs ]#line:138
    layer .num_features =layer .num_features -len (idxs )#line:139
    layer .running_mean =layer .running_mean .data .clone ()[OO00000O0000O0O0O ]#line:140
    layer .running_var =layer .running_var .data .clone ()[OO00000O0000O0O0O ]#line:141
    if layer .affine :#line:142
        layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[OO00000O0000O0O0O ])#line:143
        layer .bias =torch .nn .Parameter (layer .bias .data .clone ()[OO00000O0000O0O0O ])#line:144
    return layer ,OO0000O0O0OOOOOO0 #line:145
def prune_prelu (layer :nn .PReLU ,idxs :list ,inplace :bool =True ,dry_run :bool =False ):#line:147
    ""#line:153
    OOO00O000O0OO0O0O =0 if layer .num_parameters ==1 else len (idxs )#line:154
    if dry_run :#line:155
        return layer ,OOO00O000O0OO0O0O #line:156
    if not inplace :#line:157
        layer =deepcopy (layer )#line:158
    if layer .num_parameters ==1 :return layer ,OOO00O000O0OO0O0O #line:159
    OO0OOOOO0OO00O0O0 =[OOOO00OOOOOOO00O0 for OOOO00OOOOOOO00O0 in range (layer .num_parameters )if OOOO00OOOOOOO00O0 not in idxs ]#line:160
    layer .num_parameters =layer .num_parameters -len (idxs )#line:161
    layer .weight =torch .nn .Parameter (layer .weight .data .clone ()[OO0OOOOO0OO00O0O0 ])#line:162
    return layer ,OOO00O000O0OO0O0O #line:163
