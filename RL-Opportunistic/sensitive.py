import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time
import logging
logging.disable(30)

from simulator import AIRVIEW
from simulator import load_av_ue_info

def sac_loss(y_true, y_pred):
    qs = 0.99 * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
    return tf.reduce_sum(qs, axis=-1)

env = AIRVIEW(lambda_avg=10000, lambda_fairness=1)
av_ues_info = load_av_ue_info()
model = tf.keras.models.load_model('actor_eval_net_epoch20.h5', custom_objects={'sac_loss':sac_loss})
av_ues_idx = list(range(0, 4))
state = env.reset(av_ues_info, av_ues_idx)['user_info']

for i in range(3):
    state[i, 1] = (i+1) * 2000
    state[i, 3:53] = (i+1) * 10


state = tf.expand_dims(state, 0)
print(state)

pred = model.predict(state[..., tf.newaxis])
action = np.argmax(pred, axis=1)
print(action)
