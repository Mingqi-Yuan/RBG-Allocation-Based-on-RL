from simulator import AIRVIEW
from simulator import load_av_ue_info
import numpy as np

env = AIRVIEW()
av_ues_info = load_av_ue_info()
USER_NUM = 5

for i in range(4):
    av_begin_ue_idx = i % (len(av_ues_info) - np.random.randint(1, 17, 1)[0])
    av_ues_idx = list(range(av_begin_ue_idx, av_begin_ue_idx + USER_NUM))

    print(av_ues_idx)
