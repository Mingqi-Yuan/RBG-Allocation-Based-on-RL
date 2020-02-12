import datetime
import logging
logging.disable(30)

from model.DDPG import DDPGAgent
from model.SoftAC import SoftACAgent

from simulator import AIRVIEW
from simulator import load_av_ue_info

if __name__== '__main__':
    BATCH_SIZE = 1000
    EPOCH = 500
    MEMORY_CAPACITY = 1000000
    RGB_NUM = 17
    USER_NUM = None
    FEATURE_NUM = 21

    env = AIRVIEW()
    av_ues_info = load_av_ue_info()
    # agent = SoftACAgent(
    #     user_num=USER_NUM,
    #     feature_num=FEATURE_NUM,
    #     rgb_num=RGB_NUM,
    #     batch_size=BATCH_SIZE,
    #     replayer_capacity=MEMORY_CAPACITY,
    #     resume=False,
    #     checkpoint={
    #         'actor_eval': 'actor_eval_net_epoch90.h5',
    #         'actor_target': 'actor_target_net_epoch90.h5',
    #         'critic_eval': 'critic_eval_net_epoch90.h5',
    #         'critic_target': 'critic_target_net_epoch90.h5'
    #     })
    agent = DDPGAgent(
        user_num=USER_NUM,
        feature_num=FEATURE_NUM,
        rgb_num=RGB_NUM,
        batch_size=BATCH_SIZE,
        replayer_capacity=MEMORY_CAPACITY,
        resume=False,
        checkpoint={
            'actor_eval':'actor_eval_net_epoch90.h5',
            'actor_target':'actor_target_net_epoch90.h5',
            'critic_eval':'critic_eval_net_epoch90.h5',
            'critic_target':'critic_target_net_epoch90.h5'
        })
    performance = dict()

    if agent.resume == True:
        av_begin_ue_idx = 0
    else:
        av_begin_ue_idx = 0

    for i in range(av_begin_ue_idx, EPOCH):
        print('Epoch: {}'.format(i))
        USER_NUM = 5
        av_begin_ue_idx = i % (len(av_ues_info) - USER_NUM)
        av_ues_idx = list(range(av_begin_ue_idx, av_begin_ue_idx + USER_NUM))
        # av_ues_idx = list(range(i*5, (i+1)*5))

        # allocation based on PF
        _ = env.reset(av_ues_info, av_ues_idx)
        while True:
            state_, reward, done, info = env.step(None)
            if done:
                pf_result = env.get_result()
                performance['PF'] = performance.get('PF', [])
                performance['PF'].append((pf_result['avg_ue_rate'], pf_result['worst_ue_rate']))
                break

        # allocation based on RL
        observation = env.reset(av_ues_info, av_ues_idx)
        while True:
            action = agent.decide(observation)
            print('User index={}~{}'.format(env.av_ues_idx[0], env.av_ues_idx[-1]))
            print('Allocation result={}'.format(action['action']))
            next_observation, reward, done, _ = env.step(action['action'])
            agent.learn(observation, action['softmax_score'], reward, next_observation, done)
            if done:
                rl_result = env.get_result()
                performance['RL'] = performance.get('RL', [])
                performance['RL'].append((rl_result['avg_ue_rate'], rl_result['worst_ue_rate']))
                break
            observation = next_observation

        # clear the replayer and save the model
        # agent.replayer.clear()

        with open('record.txt', 'a') as file:
            record = '{} Epoch={}, ue_idx={}~{}, {} ues_rate:[avg={},worst={}], {} ues_rate:[avg={},worst={}]'.format(
                datetime.datetime.now(), i, env.av_ues_idx[0], env.av_ues_idx[-1],
                'PF', pf_result['avg_ue_rate'], pf_result['worst_ue_rate'],
                'RL', rl_result['avg_ue_rate'], rl_result['worst_ue_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        if i % 10 == 0 and i is not 0:
            agent.save(epoch=i)

            PF_avg, PF_worst = zip(*performance['PF'])
            RL_avg, RL_worst = zip(*performance['RL'])

            better_avg = [(r-p)/ p for p, r in zip(PF_avg, RL_avg)]
            better_avg_percent = sum(better_avg) / len(better_avg)
            better_worst = [(r-p)/ p for p, r in zip(PF_worst, RL_worst)]
            better_worst_percent = sum(better_worst) / len(better_worst)
            both_better = [1 if (a > 0 and w > 0) else 0 for a, w in zip(better_avg, better_worst)]
            both_better_percent = sum(both_better) / len(both_better)

            print("better_avg_percent:{}, better_worst_percent:{}, both_better_percent:{}".format(better_avg_percent,
                 better_worst_percent, both_better_percent))



