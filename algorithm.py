import numpy as np
import pandas as pd
import datetime
import logging
logging.disable(30)

from simulator import AIRVIEW
from simulator import load_av_ue_info

from model.DDPG import DDPGAgent

if __name__ == '__main__':
    RBG_NUM = 17
    FEATURE_NUM = 21
    TTI_SUM = 20000
    RANDOM_SEED = 7
    POSSION_AVG = 5 / 1000
    EPOCH = 200

    pf_env = AIRVIEW()
    rl_env = AIRVIEW()

    av_ues_info = load_av_ue_info()

    ddpg_agent = DDPGAgent(
        user_num=None,
        feature_num=21,
        rbg_num=17,
        replayer_capacity=20000)

    for epoch in range(EPOCH):
        print('Epoch={}'.format(epoch))
        INITIAL_USER_NUM = np.random.randint(2, 18, 10)[0]
        av_ues_idx = list(range(0, INITIAL_USER_NUM))

        pf_state = pf_env.reset(av_ues_info, av_ues_idx)
        rl_state = rl_env.reset(av_ues_info, av_ues_idx)

        tti = 0
        pf_done = False
        rl_done = False

        while (tti < TTI_SUM):
            possion_add_user = np.random.poisson(POSSION_AVG, 1)[0]

            ''' PF Allocation '''
            if pf_done == False:
                pf_next_state, pf_reward, pf_done, pf_info = pf_env.step(None, possion_add_user)

            ''' RL Allocation '''
            if rl_done == False:
                if rl_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                    rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(None, possion_add_user)
                else:
                    action = ddpg_agent.decide(rl_state)
                    rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(action['action'], possion_add_user)
                    ddpg_agent.learn(rl_state, action['softmax_score'], rl_reward, rl_next_state, done=False)

                rl_state = rl_next_state

            if pf_done == True or rl_done == True:
                ddpg_agent.learn(None, None, None, None, store=False, done=True)
                break

            tti += 1

        ''' get result '''
        pf_result = pf_env.get_result()
        rl_result = rl_env.get_result()

        ''' tti > TTI_SUM but rl_done is not True '''
        if tti == TTI_SUM:
            ddpg_agent.learn(None, None, None, None, done=True, store=False)

        ''' clear the replayer and save the model '''
        ddpg_agent.replayer.clear()

        with open('record.txt', 'a') as file:
            record = '{} Epoch={}, ue_idx={}~{}, {} ues_rate:[avg={},worst={}], {} ues_rate:[avg={},worst={}]'.format(
                datetime.datetime.now(), epoch, rl_env.av_ues_idx[0], rl_env.av_ues_idx[-1],
                'PF', pf_result['avg_ue_rate'], pf_result['worst_ue_rate'],
                'RL', rl_result['avg_ue_rate'], rl_result['worst_ue_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        if epoch % 10 == 0 and epoch is not 0:
            ddpg_agent.save(epoch=epoch)