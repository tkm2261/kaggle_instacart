import pyximport; pyximport.install()
from utils import f1
import numpy as np

print(f1(np.array([0,1,0]), np.array([0.1, 0.2, 0.1])))
