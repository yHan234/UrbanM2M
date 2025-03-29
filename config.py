import random
import os
import numpy as np
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device_ids = [0]   
# torch.cuda.set_device('cuda:{}'.format(device_ids[0]))


seed = 112
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)