import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time
import logging
logging.disable(30)

from Airview import AIRVIEW
from Airview import load_av_ue_info

# from model.DDPG import DDPGAgent
from DDPG import DDPGAgent

if __name__ == '__main__':
    RBG_NUM = 17
    FEATURE_NUM = 21
    TTI_SUM = 200
    RANDOM_SEED = 7
    POSSION_AVG = 5 / 1000
    REPLAYER_CAPACITY = 5000
    LAMBDA_AVG = 10000
    LAMBDA_FAIRNESS = 1
    EPOCH = 100
    TRAIN_INTERNAL = 10
    LR = 0.001
    SAMPLE_FRAC = 0.2

    pf_env = AIRVIEW(lambda_avg=LAMBDA_AVG, lambda_fairness=LAMBDA_FAIRNESS)
    rl_env = AIRVIEW(lambda_avg=LAMBDA_AVG, lambda_fairness=LAMBDA_FAIRNESS)

    av_ues_info = load_av_ue_info()

    ddpg_agent = DDPGAgent(
        user_num=None,
        feature_num=FEATURE_NUM,
        rbg_num=RBG_NUM,
        replayer_capacity=REPLAYER_CAPACITY,
        sample_frac=SAMPLE_FRAC,
        lr=LR)

    ddpg_agent.actor_eval_net = tf.keras.models.load_model('actor_eval_net_epoch100.h5')

    for epoch in range(EPOCH):
        time_start = time.clock()
        INITIAL_USER_START = int(epoch / TRAIN_INTERNAL)
        INITIAL_USER_NUM = np.random.randint(5, 18, 10)[0]
        av_ues_idx = list(range(INITIAL_USER_START, INITIAL_USER_START + INITIAL_USER_NUM))

        pf_state = pf_env.reset(av_ues_info, av_ues_idx)
        rl_state = rl_env.reset(av_ues_info, av_ues_idx)

        tti = 0
        while (tti < TTI_SUM):
            possion_add_user = np.random.poisson(POSSION_AVG, 1)[0]

            ''' PF Allocation '''
            pf_next_state, pf_reward, pf_done, pf_info = pf_env.step(None, possion_add_user, INITIAL_USER_START)

            ''' RL Allocation '''
            if rl_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(None, possion_add_user, INITIAL_USER_START)
            else:
                action = ddpg_agent.decide(rl_state)
                # print('TTI={},User={},Action={}'.format(tti, rl_env.bs.to_newtx_ues_idx,action['action']))
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(action['action'], possion_add_user, INITIAL_USER_START)

                rl_state = rl_next_state

            tti += 1

        ''' get result '''
        pf_result = pf_env.get_result()
        rl_result = rl_env.get_result()

        with open('eval_record.txt', 'a') as file:
            record = '{} Epoch={}, user_num={}, {} ues_rate:[avg={},five_tile_rate={}], {} ues_rate:[avg={},five_tile_rate={}]'.format(
                datetime.datetime.now(), epoch, len(rl_env.bs.ues),
                'PF', pf_result['avg_ue_rate'], pf_result['five_tile_rate'],
                'RL', rl_result['avg_ue_rate'], rl_result['five_tile_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        time_end = time.clock()
        record = pd.read_csv('eval_record.txt', header=None, sep='[=:, \]]', engine='python')
        print('Epoch={}, time cost={} s'.format(epoch, time_end - time_start))
        print('RL avg better={}'.format(record[13][record[13] < record[21]].shape[0]))
        print('RL 5%-tile better={}'.format(record[15][record[15] < record[23]].shape[0]))
        print('RL both better={}'.format(record[(record[13] < record[21]) & (record[15] < record[23])].shape[0]))
        print('PF avg better={}'.format(record[13][record[13] > record[21]].shape[0]))
        print('PF 5%-tile better={}'.format(record[15][record[15] > record[23]].shape[0]))
        print('PF both better={}'.format(record[(record[13] > record[21]) & (record[15] > record[23])].shape[0]))
        print('\n')