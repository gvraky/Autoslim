import torch #line:1
import torch .nn as nn #line:2
import typing #line:3
from functools import reduce #line:4
from operator import mul #line:5
from .import prune #line:6
from enum import IntEnum #line:7
__all__ =['PruningPlan','Dependency','DependencyGraph']#line:9
TORCH_CONV =nn .modules .conv ._ConvNd #line:11
TORCH_BATCHNORM =nn .modules .batchnorm ._BatchNorm #line:12
TORCH_PRELU =nn .PReLU #line:13
TORCH_LINEAR =nn .Linear #line:14
class O0O00000OOOO00OO0 (IntEnum ):#line:16
    CONV =0 #line:17
    BN =1 #line:18
    LINEAR =2 #line:19
    PRELU =3 #line:20
    GROUP_CONV =4 #line:21
    CONCAT =5 #line:23
    SPLIT =6 #line:24
    ELEMENTWISE =7 #line:25
def _OO0OOOO0O00O0O0OO (module ):#line:27
    if isinstance (module ,TORCH_CONV ):#line:28
        if module .groups >1 :#line:29
            return O0O00000OOOO00OO0 .GROUP_CONV #line:30
        else :#line:31
            return O0O00000OOOO00OO0 .CONV #line:32
    elif isinstance (module ,TORCH_BATCHNORM ):#line:33
        return O0O00000OOOO00OO0 .BN #line:34
    elif isinstance (module ,TORCH_PRELU ):#line:35
        return O0O00000OOOO00OO0 .PRELU #line:36
    elif isinstance (module ,TORCH_LINEAR ):#line:37
        return O0O00000OOOO00OO0 .LINEAR #line:38
    elif isinstance (module ,_O00000O00OOO0O0OO ):#line:39
        return O0O00000OOOO00OO0 .CONCAT #line:40
    elif isinstance (module ,_OO00O00O0O0O0O000 ):#line:41
        return O0O00000OOOO00OO0 .SPLIT #line:42
    else :#line:43
        return O0O00000OOOO00OO0 .ELEMENTWISE #line:44
def _O000OOOOO0O0OO0O0 (node ):#line:46
    if node .type ==O0O00000OOOO00OO0 .CONV or node .type ==O0O00000OOOO00OO0 .GROUP_CONV :#line:47
        return node .module .out_channels #line:48
    elif node .type ==O0O00000OOOO00OO0 .BN :#line:49
        return node .module .num_features #line:50
    elif node .type ==O0O00000OOOO00OO0 .LINEAR :#line:51
        return node .module .out_features #line:52
    elif node .type ==O0O00000OOOO00OO0 .PRELU :#line:53
        if node .module .num_parameters ==1 :#line:54
            return None #line:55
        else :#line:56
            return node .module .num_parameters #line:57
    else :#line:58
        return None #line:59
def _O0O0O00O000O0O0O0 (node ):#line:61
    if node .type ==O0O00000OOOO00OO0 .CONV or node .type ==O0O00000OOOO00OO0 .GROUP_CONV :#line:62
        return node .module .in_channels #line:63
    elif node .type ==O0O00000OOOO00OO0 .BN :#line:64
        return node .module .num_features #line:65
    elif node .type ==O0O00000OOOO00OO0 .LINEAR :#line:66
        return node .module .in_features #line:67
    elif node .type ==O0O00000OOOO00OO0 .PRELU :#line:68
        if node .module .num_parameters ==1 :#line:69
            return None #line:70
        else :#line:71
            return node .module .num_parameters #line:72
    else :#line:73
        return None #line:74
def _O00O0000O0000OO0O (layer ,*OO00OOOO00O0OO000 ,**OOOOO0O00O0OOO0OO ):#line:77
    return layer ,0 #line:78
def _O0OO000O0OOOOO000 (layer ,*O0OO0OOOO000O0000 ,**OO0O00O00O0O0OO0O ):#line:80
    return layer ,0 #line:81
def _OOOOO0O0OOOO00000 (layer ,*O00OO00OO00000OO0 ,**O0OO000O0OOOO00O0 ):#line:83
    return layer ,0 #line:84
class _O00000O00OOO0O0OO (nn .Module ):#line:87
    def __init__ (self ):#line:88
        super (_O00000O00OOO0O0OO ,self ).__init__ ()#line:89
        self .offsets =None #line:90
    def __repr__ (self ):#line:92
        return "_ConcatOp(%s)"%(self .offsets )#line:93
class _OO00O00O0O0O0O000 (nn .Module ):#line:95
    def __init__ (self ):#line:96
        super (_OO00O00O0O0O0O000 ,self ).__init__ ()#line:97
        self .offsets =None #line:98
    def __repr__ (self ):#line:100
        return "_SplitOP(%s)"%(self .offsets )#line:101
class _O0OOOO0000000OO00 (nn .Module ):#line:103
    def __init__ (self ):#line:104
        super (_O0OOOO0000000OO00 ,self ).__init__ ()#line:105
    def __repr__ (self ):#line:107
        return "_ElementWiseOp()"#line:108
class _O000OOOOO00O00OO0 (object ):#line:112
    def __init__ (self ,stride =1 ,reverse =False ):#line:113
        self ._stride =stride #line:114
        self .reverse =reverse #line:115
    def __call__ (self ,idxs ):#line:117
        OOOOOOO0O000O0O00 =[]#line:118
        if self .reverse ==True :#line:119
            for OO00O0000OOOO00O0 in idxs :#line:120
                OOOOOOO0O000O0O00 .append (OO00O0000OOOO00O0 //self ._stride )#line:121
                OOOOOOO0O000O0O00 =list (set (OOOOOOO0O000O0O00 ))#line:122
        else :#line:123
            for OO00O0000OOOO00O0 in idxs :#line:124
                OOOOOOO0O000O0O00 .extend (list (range (OO00O0000OOOO00O0 *self ._stride ,(OO00O0000OOOO00O0 +1 )*self ._stride )))#line:125
        return OOOOOOO0O000O0O00 #line:126
class _OO0O0000OO00OOOOO (object ):#line:128
    def __init__ (self ,offset ,reverse =False ):#line:129
        self .offset =offset #line:130
        self .reverse =reverse #line:131
    def __call__ (self ,idxs ):#line:133
        if self .reverse ==True :#line:134
            OOOO0O0OOO0O0O000 =[OO0O00O0OO000000O -self .offset [0 ]for OO0O00O0OO000000O in idxs if (OO0O00O0OO000000O >=self .offset [0 ]and OO0O00O0OO000000O <self .offset [1 ])]#line:135
        else :#line:136
            OOOO0O0OOO0O0O000 =[OOOO00OOO000OOO00 +self .offset [0 ]for OOOO00OOO000OOO00 in idxs ]#line:137
        return OOOO0O0OOO0O0O000 #line:138
class _OO0OOOOOOO0O0OOO0 (object ):#line:140
    def __init__ (self ,offset ,reverse =False ):#line:141
        self .offset =offset #line:142
        self .reverse =reverse #line:143
    def __call__ (self ,idxs ):#line:145
        if self .reverse ==True :#line:146
            O0OOOO0OOO0000000 =[OOOOOO000000OOO0O +self .offset [0 ]for OOOOOO000000OOO0O in idxs ]#line:147
        else :#line:148
            O0OOOO0OOO0000000 =[O0O000OOOO00O00OO -self .offset [0 ]for O0O000OOOO00O00OO in idxs if (O0O000OOOO00O00OO >=self .offset [0 ]and O0O000OOOO00O00OO <self .offset [1 ])]#line:149
        return O0OOOO0OOO0000000 #line:150
class OO0O000OOOOOO0OOO (object ):#line:152
    def __init__ (self ,module ,grad_fn ,node_name =None ):#line:153
        self .module =module #line:154
        self .grad_fn =grad_fn #line:155
        self .inputs =[]#line:156
        self .outputs =[]#line:157
        self .dependencies =[]#line:158
        self ._node_name =node_name #line:159
        self .type =_OO0OOOO0O00O0O0OO (module )#line:160
    @property #line:162
    def node_name (self ):#line:163
        return "%s (%s)"%(self ._node_name ,str (self .module ))if self ._node_name is not None else str (self .module )#line:164
    def add_input (self ,node ):#line:166
        if node not in self .inputs :#line:167
            self .inputs .append (node )#line:168
    def add_output (self ,node ):#line:170
        if node not in self .outputs :#line:171
            self .outputs .append (node )#line:172
    def __repr__ (self ):#line:174
        return "<Node: (%s, %s)>"%(self .node_name ,self .grad_fn )#line:175
    def __str__ (self ):#line:177
        return "<Node: (%s, %s)>"%(self .node_name ,self .grad_fn )#line:178
    def details (self ):#line:180
        OO0O0O0O00O00O00O ="<Node: (%s, %s)>\n"%(self .node_name ,self .grad_fn )#line:181
        OO0O0O0O00O00O00O +=' '*4 +'IN:\n'#line:182
        for O0OO000O00OOOOO0O in self .inputs :#line:183
            OO0O0O0O00O00O00O +=' '*8 +'%s\n'%(O0OO000O00OOOOO0O )#line:184
        OO0O0O0O00O00O00O +=' '*4 +'OUT:\n'#line:185
        for O00OOO0OO0OO0O00O in self .outputs :#line:186
            OO0O0O0O00O00O00O +=' '*8 +'%s\n'%(O00OOO0OO0OO0O00O )#line:187
        OO0O0O0O00O00O00O +=' '*4 +'DEP:\n'#line:189
        for O0000000OOOOOO0OO in self .dependencies :#line:190
            OO0O0O0O00O00O00O +=' '*8 +"%s\n"%(O0000000OOOOOO0OO )#line:191
        return OO0O0O0O00O00O00O #line:192
class Dependency (object ):#line:194
    def __init__ (self ,trigger ,handler ,broken_node :OO0O000OOOOOO0OOO ,index_transform :typing .Callable =None ):#line:195
        ""#line:202
        self .trigger =trigger #line:203
        self .handler =handler #line:204
        self .broken_node =broken_node #line:205
        self .index_transform =index_transform #line:206
    def __call__ (self ,idxs :list ,dry_run :bool =False ):#line:208
        OO00O0O0OOOO0OOO0 =self .handler (self .broken_node .module ,idxs ,dry_run =dry_run )#line:209
        return OO00O0O0OOOO0OOO0 #line:210
    def __repr__ (self ):#line:212
        return str (self )#line:213
    def __str__ (self ):#line:215
        return "<DEP: %s => %s on %s>"%("None"if self .trigger is None else self .trigger .__name__ ,self .handler .__name__ ,self .broken_node .node_name )#line:216
    def is_triggered_by (self ,pruning_fn ):#line:218
        return pruning_fn ==self .trigger #line:219
    def __eq__ (self ,other ):#line:221
        return ((self .trigger ==other .trigger )and self .handler ==other .handler and self .broken_node ==other .broken_node )#line:224
class PruningPlan (object ):#line:226
    ""#line:232
    def __init__ (self ):#line:234
        self ._plans =list ()#line:235
    def add_plan (self ,dep ,idxs ):#line:237
        self ._plans .append ((dep ,idxs ))#line:238
    @property #line:240
    def plan (self ):#line:241
        return self ._plans #line:242
    def exec (self ,dry_run =False ):#line:244
        O0O000OOOOOOOOOO0 =0 #line:245
        for OOO0OOO0O0O0OOOO0 ,OOOOOO0OOO0O00000 in self ._plans :#line:246
            _OO00O0OO0O00O000O ,O00OOO0OO00OOO0OO =OOO0OOO0O0O0OOOO0 (OOOOOO0OOO0O00000 ,dry_run =dry_run )#line:247
            O0O000OOOOOOOOOO0 +=O00OOO0OO00OOO0OO #line:248
        return O0O000OOOOOOOOOO0 #line:249
    def has_dep (self ,dep ):#line:251
        for _OOO000O00O00O000O ,_OO0O0OOO0O0OO0000 in self ._plans :#line:252
            if dep ==_OOO000O00O00O000O :#line:253
                return True #line:254
        return False #line:255
    def has_pruning_op (self ,dep ,idxs ):#line:257
        for _OOOOO000O00O0O000 ,_O0O0OOOOOOOOOOO00 in self ._plans :#line:258
            if _OOOOO000O00O0O000 .broken_node ==dep .broken_node and _OOOOO000O00O0O000 .handler ==dep .handler and _O0O0OOOOOOOOOOO00 ==idxs :#line:259
                return True #line:260
        return False #line:261
    @property #line:263
    def is_in_shortcut (self ):#line:264
        O0OO00OOOO0OOO000 =0 #line:265
        for _O00OO0OOO00O0OO00 ,_OOO0000OO0OO0O00O in self ._plans :#line:266
            if _O00OO0OOO00O0OO00 .handler .__name__ =='prune_conv':#line:267
                O0OO00OOOO0OOO000 +=1 #line:268
        if O0OO00OOOO0OOO000 >1 :#line:269
            return True #line:270
        else :#line:271
            return False #line:272
    def add_plan_and_merge (self ,dep ,idxs ):#line:274
        for OO00O0O00000OOO00 ,(_OOOO0O0OOO0000000 ,_O00O0OO0OOO000000 )in enumerate (self ._plans ):#line:275
            if _OOOO0O0OOO0000000 .broken_node ==dep .broken_node and _OOOO0O0OOO0000000 .handler ==dep .handler :#line:276
                self ._plans [OO00O0O00000OOO00 ]=(_OOOO0O0OOO0000000 ,list (set (_O00O0OO0OOO000000 +idxs )))#line:277
                return #line:278
        self .add_plan (dep ,idxs )#line:279
    def __str__ (self ):#line:281
        OOOO0O000OOOO0O00 =""#line:282
        OOOO0O000OOOO0O00 +="\n-------------\n"#line:283
        OOO000O000OOOO0OO =0 #line:284
        for OOO0OO0O0OO000O00 ,OO000OO0O0O00O000 in self ._plans :#line:285
            _O0OOO00OOO000000O ,OOO00OO00OO0OO00O =OOO0OO0O0OO000O00 (OO000OO0O0O00O000 ,dry_run =True )#line:286
            OOO000O000OOOO0OO +=OOO00OO00OO0OO00O #line:287
            OOOO0O000OOOO0O00 +="[ %s, Index=%s, NumPruned=%d]\n"%(OOO0OO0O0OO000O00 ,OO000OO0O0O00O000 ,OOO00OO00OO0OO00O )#line:288
        OOOO0O000OOOO0O00 +="%d parameters will be pruned\n"%(OOO000O000OOOO0OO )#line:289
        OOOO0O000OOOO0O00 +="-------------\n"#line:290
        return OOOO0O000OOOO0O00 #line:291
class DependencyGraph (object ):#line:294
    PRUNABLE_MODULES =(nn .modules .conv ._ConvNd ,nn .modules .batchnorm ._BatchNorm ,nn .Linear ,nn .PReLU )#line:296
    HANDLER ={O0O00000OOOO00OO0 .CONV :(prune .prune_related_conv ,prune .prune_conv ),O0O00000OOOO00OO0 .BN :(prune .prune_batchnorm ,prune .prune_batchnorm ),O0O00000OOOO00OO0 .PRELU :(prune .prune_prelu ,prune .prune_prelu ),O0O00000OOOO00OO0 .LINEAR :(prune .prune_related_linear ,prune .prune_linear ),O0O00000OOOO00OO0 .GROUP_CONV :(prune .prune_group_conv ,prune .prune_group_conv ),O0O00000OOOO00OO0 .CONCAT :(_O00O0000O0000OO0O ,_O00O0000O0000OO0O ),O0O00000OOOO00OO0 .SPLIT :(_O0OO000O0OOOOO000 ,_O0OO000O0OOOOO000 ),O0O00000OOOO00OO0 .ELEMENTWISE :(_OOOOO0O0OOOO00000 ,_OOOOO0O0OOOO00000 ),}#line:307
    OUTPUT_NODE_RULES ={}#line:308
    INPUT_NODE_RULES ={}#line:309
    for t1 in HANDLER .keys ():#line:310
        for t2 in HANDLER .keys ():#line:311
            OUTPUT_NODE_RULES [(t1 ,t2 )]=(HANDLER [t1 ][1 ],HANDLER [t2 ][0 ])#line:312
            INPUT_NODE_RULES [(t1 ,t2 )]=(HANDLER [t1 ][0 ],HANDLER [t2 ][1 ])#line:313
    def build_dependency (self ,model :torch .nn .Module ,example_inputs :torch .Tensor ,output_transform :callable =None ,verbose :bool =True ):#line:315
        self .verbose =verbose #line:316
        self ._module_to_name ={O0OO00O0O000O0O00 :OO0OOOOOO00O000O0 for (OO0OOOOOO00O000O0 ,O0OO00O0O000O0O00 )in model .named_modules ()}#line:318
        self .module_to_node ,self .output_grad_fn =self ._obtain_forward_graph (model ,example_inputs ,output_transform =output_transform )#line:320
        self ._build_dependency (self .module_to_node )#line:321
        self .update_index ()#line:322
        return self #line:323
    def update_index (self ):#line:325
        for O0OO00OOO0O00O000 ,OO0O000000000OO0O in self .module_to_node .items ():#line:326
            if OO0O000000000OO0O .type ==O0O00000OOOO00OO0 .LINEAR :#line:327
                self ._set_fc_index_transform (OO0O000000000OO0O )#line:328
            if OO0O000000000OO0O .type ==O0O00000OOOO00OO0 .CONCAT :#line:329
                self ._set_concat_index_transform (OO0O000000000OO0O )#line:330
            if OO0O000000000OO0O .type ==O0O00000OOOO00OO0 .SPLIT :#line:331
                self ._set_split_index_transform (OO0O000000000OO0O )#line:332
    def get_pruning_plan (self ,module ,pruning_fn ,idxs ):#line:334
        if isinstance (module ,TORCH_CONV )and module .groups >1 :#line:335
            pruning_fn =prune .prune_group_conv #line:336
        self .update_index ()#line:338
        O000OO00OO0000OO0 =PruningPlan ()#line:339
        O00OOOO0O0O0OOO00 =self .module_to_node [module ]#line:341
        if O00OOOO0O0O0OOO00 .grad_fn in self .output_grad_fn :#line:343
            return None #line:344
        O000OO00OO0000OO0 .add_plan (Dependency (pruning_fn ,pruning_fn ,O00OOOO0O0O0OOO00 ),idxs )#line:346
        OO0O0O00OOO000OOO =set ()#line:348
        def _O0O0O0O000O0OOOO0 (node ,fn ,indices ):#line:349
            OO0O0O00OOO000OOO .add (node )#line:350
            for OO000OOO0O000O0OO in node .dependencies :#line:351
                if OO000OOO0O000O0OO .is_triggered_by (fn ):#line:352
                    if OO000OOO0O000O0OO .index_transform is not None :#line:353
                        OO00000O000OO0OO0 =OO000OOO0O000O0OO .index_transform (indices )#line:354
                    else :#line:355
                        OO00000O000OO0OO0 =indices #line:356
                    if len (OO00000O000OO0OO0 )==0 :#line:358
                        continue #line:359
                    if OO000OOO0O000O0OO .broken_node in OO0O0O00OOO000OOO and O000OO00OO0000OO0 .has_pruning_op (OO000OOO0O000O0OO ,OO00000O000OO0OO0 ):#line:360
                        continue #line:361
                    else :#line:362
                        O000OO00OO0000OO0 .add_plan (OO000OOO0O000O0OO ,OO00000O000OO0OO0 )#line:363
                        _O0O0O0O000O0OOOO0 (OO000OOO0O000O0OO .broken_node ,OO000OOO0O000O0OO .handler ,OO00000O000OO0OO0 )#line:364
        _O0O0O0O000O0OOOO0 (O00OOOO0O0O0OOO00 ,pruning_fn ,idxs )#line:366
        O00OOOOOOO00000O0 =PruningPlan ()#line:369
        for O0O0OO0OO00OOOOOO ,idxs in O000OO00OO0000OO0 .plan :#line:370
            O00OOOOOOO00000O0 .add_plan_and_merge (O0O0OO0OO00OOOOOO ,idxs )#line:371
        return O00OOOOOOO00000O0 #line:372
    def _build_dependency (self ,module_to_node ):#line:374
        for O00OO0O0OOO0000O0 ,OOOOO00O00O0O0OO0 in module_to_node .items ():#line:375
            for OOOOO00O0O000000O in OOOOO00O00O0O0OO0 .inputs :#line:376
                O0OOOOO0O00OO00OO =self .INPUT_NODE_RULES .get ((OOOOO00O00O0O0OO0 .type ,OOOOO00O0O000000O .type ),None )#line:377
                if O0OOOOO0O00OO00OO is not None :#line:378
                    O00O0O0OO00O00000 =Dependency (trigger =O0OOOOO0O00OO00OO [0 ],handler =O0OOOOO0O00OO00OO [1 ],broken_node =OOOOO00O0O000000O )#line:379
                    OOOOO00O00O0O0OO0 .dependencies .append (O00O0O0OO00O00000 )#line:380
            for OOOOO0OOO000O0O00 in OOOOO00O00O0O0OO0 .outputs :#line:382
                OO00OOOOOO0000O0O =self .OUTPUT_NODE_RULES .get ((OOOOO00O00O0O0OO0 .type ,OOOOO0OOO000O0O00 .type ),None )#line:383
                if OO00OOOOOO0000O0O is not None :#line:384
                    O00O0O0OO00O00000 =Dependency (trigger =OO00OOOOOO0000O0O [0 ],handler =OO00OOOOOO0000O0O [1 ],broken_node =OOOOO0OOO000O0O00 )#line:385
                    OOOOO00O00O0O0OO0 .dependencies .append (O00O0O0OO00O00000 )#line:386
    def _obtain_forward_graph (self ,model ,example_inputs ,output_transform ):#line:388
        model .eval ().cpu ()#line:390
        O0OOO00OOO00000OO ={}#line:392
        O00O00OOOOOOOO0O0 ={}#line:394
        def _OOOO000OOOO0000O0 (module ,inputs ,outputs ):#line:395
            if module not in O00O00OOOOOOOO0O0 :#line:396
                O00O00OOOOOOOO0O0 [module ]=1 #line:397
            else :#line:398
                O00O00OOOOOOOO0O0 [module ]+=1 #line:399
            O0OOO00OOO00000OO [outputs .grad_fn ]=module #line:400
        OOOO00O000O0000OO =[OO000OO0O00OO0O0O .register_forward_hook (_OOOO000OOOO0000O0 )for OO000OO0O00OO0O0O in model .modules ()if isinstance (OO000OO0O00OO0O0O ,self .PRUNABLE_MODULES )]#line:402
        OOOOO0O00000OO0OO =model (example_inputs )#line:403
        for OOO0000O000O000OO in OOOO00O000O0000OO :#line:404
            OOO0000O000O000OO .remove ()#line:405
        OOO0OO00OOOOOOO00 =[O00OO0O0OO00O0O00 for (O00OO0O0OO00O0O00 ,OO00OO00OO0O00OOO )in O00O00OOOOOOOO0O0 .items ()if OO00OO00OO0O00OOO >1 ]#line:406
        OOO000000OOO000OO ={}#line:408
        O0OO000O0O0O00000 =[]#line:410
        def _O00O0OO0O0OOO000O (grad_fn ,search_final_conv =0 ):#line:412
            search_final_conv =search_final_conv #line:414
            O000O0O0O00O0OOO0 =O0OOO00OOO00000OO .get (grad_fn ,None )#line:416
            if O000O0O0O00O0OOO0 is not None and O000O0O0O00O0OOO0 in OOO000000OOO000OO and O000O0O0O00O0OOO0 not in OOO0OO00OOOOOOO00 :#line:417
                return OOO000000OOO000OO [O000O0O0O00O0OOO0 ]#line:418
            if O000O0O0O00O0OOO0 is None :#line:420
                if not hasattr (grad_fn ,'name'):#line:421
                    O000O0O0O00O0OOO0 =_O0OOOO0000000OO00 ()#line:422
                    if self .verbose :#line:423
                        print ("[Warning] Unrecognized operation: %s. It will be treated as element-wise op"%(str (grad_fn )))#line:424
                elif 'catbackward'in grad_fn .name ().lower ():#line:425
                    O000O0O0O00O0OOO0 =_O00000O00OOO0O0OO ()#line:426
                elif 'splitbackward'in grad_fn .name ().lower ():#line:427
                    O000O0O0O00O0OOO0 =_OO00O00O0O0O0O000 ()#line:428
                else :#line:429
                    O000O0O0O00O0OOO0 =_O0OOOO0000000OO00 ()#line:430
                O0OOO00OOO00000OO [grad_fn ]=O000O0O0O00O0OOO0 #line:431
            if O000O0O0O00O0OOO0 not in OOO000000OOO000OO :#line:433
                O00OOOO00O0O0O0OO =OO0O000OOOOOO0OOO (O000O0O0O00O0OOO0 ,grad_fn ,self ._module_to_name .get (O000O0O0O00O0OOO0 ,None ))#line:434
                OOO000000OOO000OO [O000O0O0O00O0OOO0 ]=O00OOOO00O0O0O0OO #line:435
            else :#line:436
                O00OOOO00O0O0O0OO =OOO000000OOO000OO [O000O0O0O00O0OOO0 ]#line:437
            if search_final_conv and grad_fn is not None and hasattr (grad_fn ,'name')and ('MkldnnConvolutionBackward'in grad_fn .name ()or 'AddmmBackward'in grad_fn .name ()):#line:439
                search_final_conv =0 #line:440
                O0OO000O0O0O00000 .append (grad_fn )#line:441
            if hasattr (grad_fn ,'next_functions'):#line:445
                for O00O00O0O0OO000OO in grad_fn .next_functions :#line:446
                    if O00O00O0O0OO000OO [0 ]is not None :#line:447
                        if hasattr (O00O00O0O0OO000OO [0 ],'name')and 'accumulategrad'in O00O00O0O0OO000OO [0 ].name ().lower ():#line:448
                            continue #line:449
                        O0OO00OOOOOO0000O =_O00O0OO0O0OOO000O (O00O00O0O0OO000OO [0 ],search_final_conv )#line:450
                        O00OOOO00O0O0O0OO .add_input (O0OO00OOOOOO0000O )#line:451
                        O0OO00OOOOOO0000O .add_output (O00OOOO00O0O0O0OO )#line:452
            return O00OOOO00O0O0O0OO #line:453
        if output_transform is not None :#line:455
            OOOOO0O00000OO0OO =output_transform (OOOOO0O00000OO0OO )#line:456
        if isinstance (OOOOO0O00000OO0OO ,(list ,tuple )):#line:458
            for OO0O0OOO0O0O0OOO0 in OOOOO0O00000OO0OO :#line:460
                if isinstance (OO0O0OOO0O0O0OOO0 ,dict ):#line:461
                    for O00O00O00OO00000O in OO0O0OOO0O0O0OOO0 :#line:462
                        if OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn is not None and hasattr (OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn ,'name')and ('MkldnnConvolutionBackward'in OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn .name ()or 'AddmmBackward'in OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn .name ()):#line:464
                            O0OO000O0O0O00000 .append (OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn )#line:465
                            _O00O0OO0O0OOO000O (OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn ,search_final_conv =0 )#line:466
                        else :#line:467
                            _O00O0OO0O0OOO000O (OO0O0OOO0O0O0OOO0 [O00O00O00OO00000O ].grad_fn ,search_final_conv =1 )#line:468
                elif isinstance (OO0O0OOO0O0O0OOO0 ,(list ,tuple )):#line:470
                    for OOO00OO0O0OO00OOO in OO0O0OOO0O0O0OOO0 :#line:471
                        if OOO00OO0O0OO00OOO .grad_fn is not None and hasattr (OOO00OO0O0OO00OOO .grad_fn ,'name')and ('MkldnnConvolutionBackward'in OOO00OO0O0OO00OOO .grad_fn .name ()or 'AddmmBackward'in OOO00OO0O0OO00OOO .grad_fn .name ()):#line:472
                            O0OO000O0O0O00000 .append (OOO00OO0O0OO00OOO .grad_fn )#line:473
                            _O00O0OO0O0OOO000O (OOO00OO0O0OO00OOO .grad_fn ,search_final_conv =0 )#line:474
                        else :#line:475
                            _O00O0OO0O0OOO000O (OOO00OO0O0OO00OOO .grad_fn ,search_final_conv =1 )#line:476
                else :#line:477
                    if OO0O0OOO0O0O0OOO0 .grad_fn is not None and hasattr (OO0O0OOO0O0O0OOO0 .grad_fn ,'name')and ('MkldnnConvolutionBackward'in OO0O0OOO0O0O0OOO0 .grad_fn .name ()or 'AddmmBackward'in OO0O0OOO0O0O0OOO0 .grad_fn .name ()):#line:478
                        O0OO000O0O0O00000 .append (OO0O0OOO0O0O0OOO0 .grad_fn )#line:479
                        _O00O0OO0O0OOO000O (OO0O0OOO0O0O0OOO0 .grad_fn ,search_final_conv =0 )#line:480
                    else :#line:481
                        _O00O0OO0O0OOO000O (OO0O0OOO0O0O0OOO0 .grad_fn ,search_final_conv =1 )#line:482
        else :#line:485
            if OOOOO0O00000OO0OO .grad_fn is not None and hasattr (OOOOO0O00000OO0OO .grad_fn ,'name')and ('MkldnnConvolutionBackward'in OOOOO0O00000OO0OO .grad_fn .name ()or 'AddmmBackward'in OOOOO0O00000OO0OO .grad_fn .name ()):#line:486
                O0OO000O0O0O00000 .append (OOOOO0O00000OO0OO .grad_fn )#line:487
                _O00O0OO0O0OOO000O (OOOOO0O00000OO0OO .grad_fn ,search_final_conv =0 )#line:488
            else :#line:489
                _O00O0OO0O0OOO000O (OOOOO0O00000OO0OO .grad_fn ,search_final_conv =1 )#line:490
        return OOO000000OOO000OO ,O0OO000O0O0O00000 #line:491
    def _set_fc_index_transform (self ,fc_node :OO0O000OOOOOO0OOO ):#line:493
        if fc_node .type !=O0O00000OOOO00OO0 .LINEAR :#line:494
            return #line:495
        O0OOOOOO000OOOOOO =set ()#line:496
        O0OO000OO0000O0OO =fc_node .module .in_features #line:497
        OOO0O0O000OO0OOOO =_O00000O0O0000OOO0 (fc_node .inputs [0 ])#line:498
        O0OOO00OOO00O0OO0 =O0OO000OO0000O0OO //OOO0O0O000OO0OOOO #line:499
        if O0OOO00OOO00O0OO0 >1 :#line:500
            for OOO0O0O0O0OO00OOO in fc_node .inputs :#line:501
                for O0O0OOO0OO00OO0O0 in fc_node .dependencies :#line:502
                    if O0O0OOO0OO00OO0O0 .broken_node ==OOO0O0O0O0OO00OOO :#line:503
                        O0O0OOO0OO00OO0O0 .index_transform =_O000OOOOO00O00OO0 (stride =O0OOO00OOO00O0OO0 ,reverse =True )#line:504
                for O0O0OOO0OO00OO0O0 in OOO0O0O0O0OO00OOO .dependencies :#line:506
                    if O0O0OOO0OO00OO0O0 .broken_node ==fc_node :#line:507
                        O0O0OOO0OO00OO0O0 .index_transform =_O000OOOOO00O00OO0 (stride =O0OOO00OOO00O0OO0 ,reverse =False )#line:508
    def _set_concat_index_transform (self ,cat_node :OO0O000OOOOOO0OOO ):#line:510
        if cat_node .type !=O0O00000OOOO00OO0 .CONCAT :#line:511
            return #line:512
        OO0O0O0O00000O0O0 =[]#line:514
        for OO00O00O0O00OO0OO in cat_node .inputs :#line:515
            OO0O0O0O00000O0O0 .append (_O00000O0O0000OOO0 (OO00O00O0O00OO0OO ))#line:516
        OO0000000O00O00O0 =[0 ]#line:518
        for O000OO00O000OOO0O in OO0O0O0O00000O0O0 :#line:519
            OO0000000O00O00O0 .append (OO0000000O00O00O0 [-1 ]+O000OO00O000OOO0O )#line:520
        cat_node .module .offsets =OO0000000O00O00O0 #line:521
        for OOO000OO0000OO00O ,O000O00000OO0O000 in enumerate (cat_node .inputs ):#line:523
            for OO0O0O00000OOO00O in cat_node .dependencies :#line:524
                if OO0O0O00000OOO00O .broken_node ==O000O00000OO0O000 :#line:525
                    OO0O0O00000OOO00O .index_transform =_OO0O0000OO00OOOOO (offset =OO0000000O00O00O0 [OOO000OO0000OO00O :OOO000OO0000OO00O +2 ],reverse =True )#line:526
            for OO0O0O00000OOO00O in O000O00000OO0O000 .dependencies :#line:528
                if OO0O0O00000OOO00O .broken_node ==cat_node :#line:529
                    OO0O0O00000OOO00O .index_transform =_OO0O0000OO00OOOOO (offset =OO0000000O00O00O0 [OOO000OO0000OO00O :OOO000OO0000OO00O +2 ],reverse =False )#line:530
    def _set_split_index_transform (self ,split_node :OO0O000OOOOOO0OOO ):#line:532
        if split_node .type !=O0O00000OOOO00OO0 .SPLIT :#line:533
            return #line:534
        OO000OO00O0000OOO =[]#line:536
        for OO000O0OO0OOOOO0O in split_node .outputs :#line:537
            OO000OO00O0000OOO .append (_OO000OOO0OO0OOOOO (OO000O0OO0OOOOO0O ))#line:538
        OO0OOOOO00OOO00O0 =[0 ]#line:540
        for OOOO00OO00OOO0O00 in OO000OO00O0000OOO :#line:541
            OO0OOOOO00OOO00O0 .append (OO0OOOOO00OOO00O0 [-1 ]+OOOO00OO00OOO0O00 )#line:542
        split_node .module .offsets =OO0OOOOO00OOO00O0 #line:543
        for OOOOOOO0O00O00OO0 ,OO000O0O00OO000OO in enumerate (split_node .outputs ):#line:544
            for OO0O0000O0OOOOO0O in split_node .dependencies :#line:545
                if OO0O0000O0OOOOO0O .broken_node ==OO000O0O00OO000OO :#line:546
                    OO0O0000O0OOOOO0O .index_transform =_OO0OOOOOOO0O0OOO0 (offset =OO0OOOOO00OOO00O0 [OOOOOOO0O00O00OO0 :OOOOOOO0O00O00OO0 +2 ],reverse =False )#line:547
            for OO0O0000O0OOOOO0O in OO000O0O00OO000OO .dependencies :#line:549
                if OO0O0000O0OOOOO0O .broken_node ==split_node :#line:550
                    OO0O0000O0OOOOO0O .index_transform =_OO0OOOOOOO0O0OOO0 (offset =OO0OOOOO00OOO00O0 [OOOOOOO0O00O00OO0 :OOOOOOO0O00O00OO0 +2 ],reverse =True )#line:551
def _O00000O0O0000OOO0 (node ):#line:553
    OOO0O0O0OOO00OO00 =_O000OOOOO0O0OO0O0 (node )#line:554
    if OOO0O0O0OOO00OO00 is None :#line:555
        OOO0O0O0OOO00OO00 =0 #line:556
        for O00O0OOO0O00OOOOO in node .inputs :#line:557
            if node .type ==O0O00000OOOO00OO0 .CONCAT :#line:558
                OOO0O0O0OOO00OO00 +=_O00000O0O0000OOO0 (O00O0OOO0O00OOOOO )#line:559
            else :#line:560
                OOO0O0O0OOO00OO00 =_O00000O0O0000OOO0 (O00O0OOO0O00OOOOO )#line:561
    return OOO0O0O0OOO00OO00 #line:562
def _OO000OOO0OO0OOOOO (node ):#line:564
    O0O0OOO0O0O0O0O00 =_O0O0O00O000O0O0O0 (node )#line:565
    if O0O0OOO0O0O0O0O00 is None :#line:566
        O0O0OOO0O0O0O0O00 =0 #line:567
        for OOOOO000OO0OO0O0O in node .outputs :#line:568
            if node .type ==O0O00000OOOO00OO0 .SPLIT :#line:569
                O0O0OOO0O0O0O0O00 +=_OO000OOO0OO0OOOOO (OOOOO000OO0OO0O0O )#line:570
            else :#line:571
                O0O0OOO0O0O0O0O00 =_OO000OOO0OO0OOOOO (OOOOO000OO0OO0O0O )#line:572
    return O0O0OOO0O0O0O0O00 