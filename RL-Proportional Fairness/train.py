import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import time
import logging
logging.disable(30)

from Airview import AIRVIEW
from Airview import load_av_ue_info

from DDPG import DDPGAgent
from SoftAC import SoftACAgent

if __name__ == '__main__':
    RBG_NUM = 17
    FEATURE_NUM = 57
    TTI_SUM = 1000
    RANDOM_SEED = 7
    POSSION_AVG = 5 / 1000
    REPLAYER_CAPACITY = 5000
    LAMBDA_AVG = 10000
    LAMBDA_FAIRNESS = 10000
    EPOCH = 1000
    INTERNAL = 1
    LR = 0.0005
    SAMPLE_FRAC = 0.6

    pf_env = AIRVIEW(lambda_avg=LAMBDA_AVG, lambda_fairness=LAMBDA_FAIRNESS)
    rl_env = AIRVIEW(lambda_avg=LAMBDA_AVG, lambda_fairness=LAMBDA_FAIRNESS)

    av_ues_info = load_av_ue_info()

    ddpg_agent = DDPGAgent(
        user_num=None,
        feature_num=FEATURE_NUM,
        rbg_num=RBG_NUM,
        replayer_capacity=REPLAYER_CAPACITY,
        sample_frac=SAMPLE_FRAC,
        lr=LR,
        resume=False
    )

    ddpg_agent.actor_eval_net.summary()
    """ resume """
    # ddpg_agent.actor_eval_net = tf.keras.models.load_model('./actor_eval_net_epoch35.h5')

    # softac_agent = SoftACAgent(
    #     user_num=None,
    #     feature_num=FEATURE_NUM,
    #     rbg_num=RBG_NUM,
    #     replayer_capacity=REPLAYER_CAPACITY,
    #     sample_frac=SAMPLE_FRAC,
    #     lr=LR
    # )

    for epoch in range(EPOCH):
        time_start = time.clock()
        INITIAL_USER_START = int((epoch+1) / INTERNAL)
        INITIAL_USER_NUM = np.random.randint(10, 18, 10)[0]
        av_ues_idx = list(range(INITIAL_USER_START, INITIAL_USER_START + INITIAL_USER_NUM))

        pf_state = pf_env.reset(av_ues_info, av_ues_idx)
        rl_state = rl_env.reset(av_ues_info, av_ues_idx)

        tti = 0
        while (tti < TTI_SUM):
            POSSION_ADD_USER = np.random.poisson(POSSION_AVG, 1)[0]

            ''' PF Allocation '''
            pf_next_state, pf_reward, pf_done, pf_info = pf_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)

            ''' RL Allocation '''
            if rl_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)
            else:
                action = ddpg_agent.decide(rl_state)
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(action['action'], POSSION_ADD_USER, INITIAL_USER_START)
                ddpg_agent.learn(rl_state, action['softmax_score'], rl_reward, rl_next_state, done=False, store=True)

                rl_state = rl_next_state

            tti += 1

        ''' get result '''
        pf_result = pf_env.get_result()
        rl_result = rl_env.get_result()

        for i in range(5):
            ''' model learning '''
            ddpg_agent.learn(None, None, None, None, done=True, store=False)

        ''' clear the replayer and save the model '''
        ddpg_agent.replayer.clear()

        train_record = './rl2pf_train.txt'
        with open(train_record, 'a') as file:
            record = 'Epoch={}, user_num={}, {} ues_rate:[avg={},five_tile_rate={}], {} ues_rate:[avg={},five_tile_rate={}]'.format(
                epoch+1, len(rl_env.bs.ues),
                'PF', pf_result['avg_ue_rate'], pf_result['five_tile_rate'],
                'RL', rl_result['avg_ue_rate'], rl_result['five_tile_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        time_end = time.clock()
        record = pd.read_csv(train_record, header=None, sep='[=:, \]]', engine='python')
        print('Epoch={}, time cost={} s'.format(epoch+1, time_end - time_start))
        print('User start index={}'.format(INITIAL_USER_START))
        print('RL avg better={}'.format(record[9][record[9] < record[17]].shape[0]))
        print('RL 5%-tile better={}'.format(record[11][record[11] < record[19]].shape[0]))
        print('RL both better={}'.format(record[(record[9] < record[17]) & (record[11] < record[19])].shape[0]))

        print('PF avg better={}'.format(record[9][record[9] > record[17]].shape[0]))
        print('PF 5%-tile better={}'.format(record[11][record[11] > record[19]].shape[0]))
        print('PF both better={}'.format(record[(record[9] > record[17]) & (record[11] > record[19])].shape[0]))
        print('\n')

        ddpg_agent.save(epoch=epoch + 1)
