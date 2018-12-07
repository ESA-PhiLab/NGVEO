#Take a pytorch variable and make numpy
import torch
from torch.autograd import Variable
import numpy as np

def torch_to_np(var):

    if type(var) in [np.ndarray, np.array, int, float]:
        return var

    old_var = var
    try:
        var = var.cpu()
    except:
        None
    var = var.data
    var = var.numpy()

    #Delete old_var explicitly to ensure GPU-memmory is releases
    del old_var
    return var


#Function that returns the GPU number of a variable (or False if on CPU)
def gpu_no_of_var(var):
    try:
        is_cuda = next(var.parameters()).is_cuda
    except:
        is_cuda = var.is_cuda

    if is_cuda:
        try:
            return next(var.parameters()).get_device()
        except:
            return var.get_device()
    else:
        return False



#Take a numpy variable and make a pytorch variable
def np_to_torch(var, gpu_no=False, volatile = False):
    # If input is list we do this for all elements
    if type(var) == type([]):
        out = []
        for v in var:
            out.append(np_to_torch(v))
        return out

    #Make numpy object
    if type(var) in [type(0),type(0.0), np.float64]:
        var = np.array([var])
    #Make tensor
    if type(var) in [np.ndarray]:
        var = torch.from_numpy(var.astype('float32')).float()
    #Make Variable
    if 'Tensor' in str(type(var)):
        var = Variable(var,volatile=volatile)
    #Put on GPU
    if type(gpu_no) == int:
        var = var.cuda(int(gpu_no))
    return var