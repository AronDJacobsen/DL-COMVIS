import torch

class DCGAN():
    
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        raise NotImplementedError("Please implement DCGAN first...")