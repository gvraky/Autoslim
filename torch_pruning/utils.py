from .dependency import O0OOO00O0O0OOO0O0, O0OOOOOOO0O0O0OO0, O0OOO00O0O0O0O0O0, OO000OOOO000O0OOO

def count_prunable_params(module):
    if isinstance( module, ( O0OOO00O0O0OOO0O0, OO000OOOO000O0OOO) ):
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()
        return num_params
    elif isinstance( module, O0OOOOOOO0O0O0OO0 ):
        num_params = module.running_mean.numel() + module.running_var.numel()
        if module.affine:
            num_params+= module.weight.numel() + module.bias.numel()
        return num_params
    elif isinstance( module, O0OOO00O0O0O0O0O0 ):
        if len( module.weight )==1:
            return 0
        else:
            return module.weight.numel
    else:
        return 0

def count_prunable_channels(module):
    if isinstance( module, O0OOO00O0O0OOO0O0 ):
        return module.weight.shape[0]
    elif isinstance( module, OO000OOOO000O0OOO ):
        return module.out_features
    elif isinstance( module, O0OOOOOOO0O0O0OO0 ):
        return module.num_features
    elif isinstance( module, O0OOO00O0O0O0O0O0 ):
        if len( module.weight )==1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0

def count_params(module):
    return sum([ p.numel() for p in module.parameters() ])
