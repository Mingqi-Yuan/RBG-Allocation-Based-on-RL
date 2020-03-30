import time
import pandas as pd
import logging
logging.disable(30)
import tensorflow as tf
import numpy as np

from Airview import AIRVIEW
from Airview import load_av_ue_info
from Opportunistic import OPPORTUNISTIC

from SoftAC import SoftACAgent
from DDPG import DDPGAgent

def user_info_threshold(user_info, threshold_min, threshold_max):
    filtered_user_info = []
    for i in range(len(user_info)):
        if (user_info[i]['buffer'] > threshold_min) and (user_info[i]['buffer'] < threshold_max):
            filtered_user_info.append(user_info[i])
    return filtered_user_info

if __name__ == '__main__':
    RBG_NUM = 17
    FEATURE_NUM = 57
    TTI_SUM = 500
    RANDOM_SEED = 7
    POSSION_AVG = 5 / 1000
    REPLAYER_CAPACITY = 5000
    LAMBDA_AVG = 10000
    LAMBDA_FAIRNESS = 1
    EPOCH = 100
    INTERNAL = 1
    LR = 0.001
    SAMPLE_FRAC = 0.2

    op_env = OPPORTUNISTIC()
    rl_env = AIRVIEW(LAMBDA_AVG, LAMBDA_FAIRNESS)

    # softac_agent = SoftACAgent(
    #     user_num=None,
    #     feature_num=FEATURE_NUM,
    #     rbg_num=RBG_NUM,
    #     replayer_capacity=REPLAYER_CAPACITY,
    #     sample_frac=SAMPLE_FRAC,
    #     lr=LR
    # )

    ddpg_agent = DDPGAgent(
        user_num=None,
        feature_num=FEATURE_NUM,
        rbg_num=RBG_NUM,
        replayer_capacity=REPLAYER_CAPACITY,
        sample_frac=SAMPLE_FRAC,
        lr=LR
    )

    ''' evaluation '''
    ddpg_agent.actor_eval_net = tf.keras.models.load_model('./actor_eval_net_epoch180.h5')

    av_ues_info = load_av_ue_info()
    av_ues_info = user_info_threshold(av_ues_info, threshold_min=10e+5, threshold_max=10e+6)

    for epoch in range(EPOCH):
        time_start = time.clock()
        INITIAL_USER_START = int((epoch+1) / INTERNAL)
        INITIAL_USER_NUM = 10
        av_ues_idx = list(range(INITIAL_USER_START, INITIAL_USER_START + INITIAL_USER_NUM))
        op_state = op_env.reset(av_ues_info, av_ues_idx)
        rl_state = rl_env.reset(av_ues_info, av_ues_idx)

        tti = 0
        while (tti < TTI_SUM):
            POSSION_ADD_USER = np.random.poisson(POSSION_AVG, 1)[0]

            ''' OP Allocation '''
            if op_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                op_next_state, op_reward, op_done, op_info = op_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)
            else:
                op_action = op_env.action(op_state)
                op_next_state, op_reward, op_done, op_info = op_env.step(op_action['action'], POSSION_ADD_USER, INITIAL_USER_START)
                op_state = op_next_state

            ''' RL Allocation '''
            if rl_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)
            else:
                start = time.clock()
                rl_action = ddpg_agent.decide(rl_state)
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(rl_action['action'], POSSION_ADD_USER, INITIAL_USER_START)
                rl_state = rl_next_state

            tti += 1

        ''' get result '''
        op_result = op_env.get_result()
        rl_result = rl_env.get_result()

        train_record = './rl2opportunistic_eval_fullbuffer.txt'
        with open(train_record, 'a') as file:
            record = 'Epoch={}, user_num={}, {} ues_rate:[avg={}], {} ues_rate:[avg={}]'.format(
                epoch+1, len(rl_env.bs.ues),
                'OP', op_result['avg_ue_rate'],
                'RL', rl_result['avg_ue_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        time_end = time.clock()
        record = pd.read_csv(train_record, header=None, sep='[=:, \]]', engine='python')
        print('Epoch={}, time cost={}s'.format(epoch+1, time_end - time_start))
        print('RL avg better={}'.format(record[9][record[9] < record[15]].shape[0]))
        print('OP avg better={}'.format(record[9][record[9] > record[15]].shape[0]))
        print('\n')

