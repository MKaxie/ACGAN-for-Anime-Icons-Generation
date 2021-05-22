import torch
import numpy as np
from model.ACGAN import ICONGAN
from utils.configs import Configs


args = Configs()
icongan = ICONGAN(args)
icongan.train()


test_labels = np.random.randint(0,2,(25,26)).astype(np.float32)

tests = torch.tensor(test_labels)

icongan.generate(tests)





