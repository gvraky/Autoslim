import torch #line:1
import torch .nn as nn #line:2
import numpy as np #line:3
from itertools import chain #line:4
from .dependency import *#line:5
from .import prune #line:6
import math #line:7
from scipy .spatial import distance #line:8
__all__ =['Autoslim']#line:10
class Autoslim (object ):#line:12
    def __init__ (self ,model ,inputs ,compression_ratio ):#line:13
        self .model =model #line:14
        self .inputs =inputs #line:15
        self .compression_ratio =compression_ratio #line:16
        self .DG =DependencyGraph ()#line:17
        self .DG .build_dependency (model ,example_inputs =inputs )#line:19
        self .model_modules =list (model .modules ())#line:20
    def index_of_layer (self ):#line:22
        O00000O00OOO00OO0 ={}#line:23
        for OO00OOO0OO00OOO00 ,O00000O00O000OO00 in enumerate (self .model_modules ):#line:24
            if isinstance (O00000O00O000OO00 ,nn .modules .conv ._ConvNd ):#line:25
                O00000O00OOO00OO0 [OO00OOO0OO00OOO00 ]=O00000O00O000OO00 #line:26
        return O00000O00OOO00OO0 #line:27
    def base_prunging (self ,pruning_func ):#line:32
        O00000OOO000OOOOO ={}#line:33
        for OO0OOOO0O0O0O000O ,O0000OO0OO0OO0O0O in enumerate (self .model_modules ):#line:34
            if isinstance (O0000OO0OO0OO0O0O ,nn .modules .conv ._ConvNd ):#line:35
                O00000OOO000OOOOO [OO0OOOO0O0O0O000O ]=O0000OO0OO0OO0O0O .out_channels #line:36
        for OO0OOOO0O0O0O000O ,O0000OO0OO0OO0O0O in enumerate (self .model_modules ):#line:37
            if isinstance (O0000OO0OO0OO0O0O ,nn .modules .conv ._ConvNd ):#line:39
                O00OOOO00O0OOO000 =weight .shape [0 ]#line:40
                if isinstance (O0000OO0OO0OO0O0O ,nn .modules .conv ._ConvTransposeMixin ):#line:42
                    O00OOOO00O0OOO000 =weight .shape [1 ]#line:43
                O0O0OO0000OO0O000 =pruning_func (O0000OO0OO0OO0O0O )#line:45
                if O0000OO0OO0OO0O0O .out_channels ==O00000OOO000OOOOO [OO0OOOO0O0O0O000O ]:#line:47
                    O000000OO00000O00 =self .DG .get_pruning_plan (O0000OO0OO0OO0O0O ,prune .prune_conv ,idxs =O0O0OO0000OO0O000 )#line:48
                    if O000000OO00000O00 :#line:49
                        if prune_shortcut ==1 :#line:50
                            O000000OO00000O00 .exec ()#line:51
                    else :#line:52
                        if not O000000OO00000O00 .is_in_shortcut :#line:53
                            O000000OO00000O00 .exec ()#line:54
    def fpgm_pruning (self ,norm_rate =1 ,dist_type ='l2',layer_compression_ratio =None ,prune_shortcut =1 ):#line:56
        O0O00000O00OO0O0O ={}#line:58
        for OO0OOOOO00OO00000 ,OO000O00OOO0OO00O in enumerate (self .model_modules ):#line:59
            if isinstance (OO000O00OOO0OO00O ,nn .modules .conv ._ConvNd ):#line:60
                O0O00000O00OO0O0O [OO0OOOOO00OO00000 ]=OO000O00OOO0OO00O .out_channels #line:61
        if layer_compression_ratio is None and prune_shortcut ==1 :#line:62
            layer_compression_ratio ={}#line:63
            OO0OO00OO00OOOO0O =self .compression_ratio #line:64
            O000O000OO0OOOO0O =(1 -OO0OO00OO00OOOO0O )/4 if OO0OO00OO00OOOO0O >=0.43 else OO0OO00OO00OOOO0O /4 #line:66
            OO0000O00O00OOO00 =[OO0OO00OO00OOOO0O -O000O000OO0OOOO0O *3 ,OO0OO00OO00OOOO0O -O000O000OO0OOOO0O *2 ,OO0OO00OO00OOOO0O -O000O000OO0OOOO0O ,OO0OO00OO00OOOO0O ,OO0OO00OO00OOOO0O +O000O000OO0OOOO0O ,OO0OO00OO00OOOO0O +O000O000OO0OOOO0O *2 ,OO0OO00OO00OOOO0O +O000O000OO0OOOO0O *3 ]#line:68
            O00OO000OOO000OOO =0 #line:69
            for OO0OOOOO00OO00000 ,OO000O00OOO0OO00O in enumerate (self .model_modules ):#line:70
                if isinstance (OO000O00OOO0OO00O ,nn .modules .conv ._ConvNd ):#line:71
                    layer_compression_ratio [OO0OOOOO00OO00000 ]=0 #line:72
                    O00OO000OOO000OOO +=1 #line:73
            O0O000OOO0OOOOO00 =O00OO000OOO000OOO /7 #line:74
            OO00OO0O00O0000O0 =0 #line:75
            for OO0OOOOO00OO00000 ,OO000O00OOO0OO00O in enumerate (self .model_modules ):#line:76
                if isinstance (OO000O00OOO0OO00O ,nn .modules .conv ._ConvNd ):#line:77
                    layer_compression_ratio [OO0OOOOO00OO00000 ]=OO0000O00O00OOO00 [math .floor (OO00OO0O00O0000O0 /O0O000OOO0OOOOO00 )]#line:78
                    OO00OO0O00O0000O0 +=1 #line:79
        for OO0OOOOO00OO00000 ,OO000O00OOO0OO00O in enumerate (self .model_modules ):#line:82
            if isinstance (OO000O00OOO0OO00O ,nn .modules .conv ._ConvNd ):#line:84
                OOO0OOOO00O0O0O0O =OO000O00OOO0OO00O .weight .detach ().cuda ()#line:85
                OO000O0O000OOO00O =OOO0OOOO00O0O0O0O .view (OOO0OOOO00O0O0O0O .size ()[0 ],-1 )#line:87
                OO0OO0O0O0OOOOOO0 =OOO0OOOO00O0O0O0O .size ()[0 ]#line:88
                if isinstance (OO000O00OOO0OO00O ,nn .modules .conv ._ConvTransposeMixin ):#line:90
                    OO000O0O000OOO00O =OOO0OOOO00O0O0O0O .view (OOO0OOOO00O0O0O0O .size ()[1 ],-1 )#line:91
                    OO0OO0O0O0OOOOOO0 =OOO0OOOO00O0O0O0O .size ()[1 ]#line:92
                if layer_compression_ratio and OO0OOOOO00OO00000 in layer_compression_ratio :#line:94
                    O0OO0O0000000O00O =int (OO0OO0O0O0OOOOOO0 *layer_compression_ratio [OO0OOOOO00OO00000 ])#line:95
                else :#line:97
                    O0OO0O0000000O00O =int (OO0OO0O0O0OOOOOO0 *self .compression_ratio )#line:98
                O000000000O0O000O =int (OO0OO0O0O0OOOOOO0 *(1 -norm_rate ))#line:100
                if dist_type =="l2"or "cos":#line:102
                    OOOOOO000OOOOOOO0 =torch .norm (OO000O0O000OOO00O ,2 ,1 )#line:103
                    O0000O0OOOO000O0O =OOOOOO000OOOOOOO0 .cpu ().numpy ()#line:104
                elif dist_type =="l1":#line:105
                    OOOOOO000OOOOOOO0 =torch .norm (OO000O0O000OOO00O ,1 ,1 )#line:106
                    O0000O0OOOO000O0O =OOOOOO000OOOOOOO0 .cpu ().numpy ()#line:107
                OOOOO00OO00O00O0O =[]#line:109
                OOOO000O000OO0O00 =[]#line:110
                OOOO000O000OO0O00 =O0000O0OOOO000O0O .argsort ()[O000000000O0O000O :]#line:111
                OOOOO00OO00O00O0O =O0000O0OOOO000O0O .argsort ()[:O000000000O0O000O ]#line:112
                OOOOOOO00O0O0000O =torch .LongTensor (OOOO000O000OO0O00 ).cuda ()#line:115
                O000O00OO0O0OOOO0 =torch .index_select (OO000O0O000OOO00O ,0 ,OOOOOOO00O0O0000O ).cpu ().numpy ()#line:117
                if dist_type =="l2"or "l1":#line:120
                    O000000O0OO00OOOO =distance .cdist (O000O00OO0O0OOOO0 ,O000O00OO0O0OOOO0 ,'euclidean')#line:121
                elif dist_type =="cos":#line:122
                    O000000O0OO00OOOO =1 -distance .cdist (O000O00OO0O0OOOO0 ,O000O00OO0O0OOOO0 ,'cosine')#line:123
                O0O0OO0O0000000O0 =np .sum (np .abs (O000000O0OO00OOOO ),axis =0 )#line:127
                O00O0OOOO0O00O0OO =O0O0OO0O0000000O0 .argsort ()[O0OO0O0000000O00O :]#line:130
                O0OO0O000O00O0O00 =O0O0OO0O0000000O0 .argsort ()[:O0OO0O0000000O00O ]#line:131
                OOOOO00OOO00O0OO0 =[OOOO000O000OO0O00 [O0000O00O0O0O00O0 ]for O0000O00O0O0O00O0 in O0OO0O000O00O0O00 ]#line:132
                if OO000O00OOO0OO00O .out_channels ==O0O00000O00OO0O0O [OO0OOOOO00OO00000 ]:#line:134
                    O0OOO00OOO00000O0 =self .DG .get_pruning_plan (OO000O00OOO0OO00O ,prune .prune_conv ,idxs =OOOOO00OOO00O0OO0 )#line:135
                    if O0OOO00OOO00000O0 :#line:136
                        if prune_shortcut ==1 :#line:137
                            O0OOO00OOO00000O0 .exec ()#line:138
                    else :#line:139
                        if not O0OOO00OOO00000O0 .is_in_shortcut :#line:140
                            O0OOO00OOO00000O0 .exec ()#line:141
    def l1_norm_pruning (self ,global_pruning =False ,layer_compression_ratio =None ,prune_shortcut =1 ):#line:145
        O0O0OOO0O0O000O00 ={}#line:147
        for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:148
            if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:149
                O0O0OOO0O0O000O00 [O0OOO000OOO00O000 ]=OOO0OO0O00O000OO0 .out_channels #line:150
        if global_pruning :#line:153
            OO00O00O000OO0OOO =[]#line:154
            for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:155
                if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:156
                    O0O00000000OO00OO =OOO0OO0O00O000OO0 .weight .detach ().cpu ().numpy ()#line:157
                    OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(1 ,2 ,3 ))#line:158
                    if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvTransposeMixin ):#line:159
                        OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(0 ,2 ,3 ))#line:160
                    OO00O00O000OO0OOO .append (OO0000O00OO00O0OO .tolist ())#line:161
            OO00O00O000OO0OOO =list (chain .from_iterable (OO00O00O000OO0OOO ))#line:163
            O000OOOO0OOOO00OO =len (OO00O00O000OO0OOO )#line:164
            OO00O00O000OO0OOO .sort ()#line:165
            OO0000000000O000O =int (O000OOOO0OOOO00OO *self .compression_ratio )#line:166
            OO0000O0OO00OO0OO =OO00O00O000OO0OOO [OO0000000000O000O ]#line:167
            for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:169
                if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:170
                    O0O00000000OO00OO =OOO0OO0O00O000OO0 .weight .detach ().cpu ().numpy ()#line:171
                    OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(1 ,2 ,3 ))#line:172
                    if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvTransposeMixin ):#line:174
                        OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(0 ,2 ,3 ))#line:175
                    OOO0O000000O0O00O =len (OO0000O00OO00O0OO [OO0000O00OO00O0OO <OO0000O0OO00OO0OO ])#line:178
                    OOO0OO00OO00OO0O0 =np .argsort (OO0000O00OO00O0OO )[:OOO0O000000O0O00O ].tolist ()#line:180
                    if OOO0OO0O00O000OO0 .out_channels ==O0O0OOO0O0O000O00 [O0OOO000OOO00O000 ]:#line:181
                        O00O000O0O000OO0O =self .DG .get_pruning_plan (OOO0OO0O00O000OO0 ,prune .prune_conv ,idxs =OOO0OO00OO00OO0O0 )#line:182
                        if O00O000O0O000OO0O :#line:183
                            O00O000O0O000OO0O .exec ()#line:184
        else :#line:187
            '''
            自定义压缩时：

            剪跳连层与不剪都可以

            全自动化压缩时：

            剪跳连层：分级剪枝
            不剪跳连层：按照用户指定的阈值剪枝

            '''#line:198
            if layer_compression_ratio is None and prune_shortcut ==1 :#line:201
                layer_compression_ratio ={}#line:202
                OOO000O0O00000000 =self .compression_ratio #line:203
                OOO00O000000OO0O0 =(1 -OOO000O0O00000000 )/4 #line:204
                O00O0000OO0OO00O0 =[OOO000O0O00000000 -OOO00O000000OO0O0 *3 ,OOO000O0O00000000 -OOO00O000000OO0O0 *2 ,OOO000O0O00000000 -OOO00O000000OO0O0 ,OOO000O0O00000000 ,OOO000O0O00000000 +OOO00O000000OO0O0 ,OOO000O0O00000000 +OOO00O000000OO0O0 *2 ,OOO000O0O00000000 +OOO00O000000OO0O0 *3 ]#line:205
                OO0O000O0O0OOO0OO =0 #line:206
                for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:207
                    if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:208
                        layer_compression_ratio [O0OOO000OOO00O000 ]=0 #line:210
                        OO0O000O0O0OOO0OO +=1 #line:211
                OO00O0OOOOO000OOO =OO0O000O0O0OOO0OO /7 #line:212
                O0000O000OO0OO000 =0 #line:213
                for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:214
                    if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:215
                        layer_compression_ratio [O0OOO000OOO00O000 ]=O00O0000OO0OO00O0 [math .floor (O0000O000OO0OO000 /OO00O0OOOOO000OOO )]#line:216
                        O0000O000OO0OO000 +=1 #line:217
            for O0OOO000OOO00O000 ,OOO0OO0O00O000OO0 in enumerate (self .model_modules ):#line:220
                if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvNd ):#line:222
                    O0O00000000OO00OO =OOO0OO0O00O000OO0 .weight .detach ().cpu ().numpy ()#line:224
                    OOO0OO0O0O00O00O0 =O0O00000000OO00OO .shape [0 ]#line:225
                    OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(1 ,2 ,3 ))#line:226
                    if isinstance (OOO0OO0O00O000OO0 ,nn .modules .conv ._ConvTransposeMixin ):#line:228
                        OO0000O00OO00O0OO =np .sum (np .abs (O0O00000000OO00OO ),axis =(0 ,2 ,3 ))#line:229
                        OOO0OO0O0O00O00O0 =O0O00000000OO00OO .shape [1 ]#line:230
                    if layer_compression_ratio and O0OOO000OOO00O000 in layer_compression_ratio :#line:233
                        OOO0O000000O0O00O =int (OOO0OO0O0O00O00O0 *layer_compression_ratio [O0OOO000OOO00O000 ])#line:234
                    else :#line:236
                        OOO0O000000O0O00O =int (OOO0OO0O0O00O00O0 *self .compression_ratio )#line:237
                    OOO0OO00OO00OO0O0 =np .argsort (OO0000O00OO00O0OO )[:OOO0O000000O0O00O ].tolist ()#line:239
                    if OOO0OO0O00O000OO0 .out_channels ==O0O0OOO0O0O000O00 [O0OOO000OOO00O000 ]:#line:241
                        O00O000O0O000OO0O =self .DG .get_pruning_plan (OOO0OO0O00O000OO0 ,prune .prune_conv ,idxs =OOO0OO00OO00OO0O0 )#line:243
                        if O00O000O0O000OO0O :#line:244
                            if prune_shortcut ==1 :#line:245
                                O00O000O0O000OO0O .exec ()#line:246
                            else :#line:247
                                if not O00O000O0O000OO0O .is_in_shortcut :#line:248
                                    O00O000O0O000OO0O .exec ()#line:249
if __name__ =="__main__":#line:252
    from resnet_small import resnet_small #line:253
    OOO0O00OOOO00OO0O =resnet_small ()#line:254
    OOOOO00OO00OO0OO0 =Autoslim (OOO0O00OOOO00OO0O ,inputs =torch .randn (1 ,3 ,224 ,224 ),compression_ratio =0.5 )#line:255
    OOOOO00OO00OO0OO0 .l1_norm_pruning ()#line:256
    print (OOO0O00OOOO00OO0O )#line:257
