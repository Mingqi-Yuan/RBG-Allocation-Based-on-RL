import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.DDPG import DDPGAgent
from simulator import AIRVIEW, load_av_ue_info

if __name__ == '__main__':
    EPOCH = 300
    RGB_NUM = 17
    USER_NUM = None
    FEATURE_NUM = 21

    av_ues_info = load_av_ue_info()
    env = AIRVIEW()

    agent = DDPGAgent(USER_NUM, FEATURE_NUM, RGB_NUM)
    agent.actor_eval_net = tf.keras.models.load_model('snapshots/actor_target_net.h5')
    performance = dict()

    for i in range(EPOCH):
        print('Epoch: ', i)
        USER_NUM = np.random.randint(1, 17, 1)[0]
        av_begin_ue_idx = i % (len(av_ues_info) - USER_NUM)
        av_ues_idx = list(range(av_begin_ue_idx, av_begin_ue_idx + USER_NUM))

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
            print('Allocation result: ', action['action'])
            next_observation, reward, done, _ = env.step(action['action'])
            if done:
                rl_result = env.get_result()
                performance['RL'] = performance.get('RL', [])
                performance['RL'].append((rl_result['avg_ue_rate'], rl_result['worst_ue_rate']))
                break
            observation = next_observation

    PF_avg, PF_worst = zip(*performance['PF'])
    RL_avg, RL_worst = zip(*performance['RL'])

    better_avg_percent_record = [(r - p) / p for p, r in zip(PF_avg, RL_avg)]
    better_worst_percent_record = [(r - p) / p for p, r in zip(PF_worst, RL_worst)]
    both_better_record = [1 if a > 0 and w > 0 else 0 for a, w in
                          zip(better_avg_percent_record, better_worst_percent_record)]

    better_avg_percent = sum(better_avg_percent_record) / len(better_avg_percent_record)
    better_worst_percent = sum(better_worst_percent_record) / len(better_worst_percent_record)
    both_better_percent = sum(both_better_record) / len(both_better_record)

    np.save('better_avg_percent.npy', better_avg_percent_record)
    np.save('better_worst_percent.npy', better_worst_percent_record)

    print("better_avg_percent:{}, better_worst_percent:{}, both_better_percent{}".format(
        better_avg_percent, better_worst_percent, both_better_percent))

    df = pd.DataFrame()
    df['RL_avg'] = RL_avg
    df['RL_worst'] = RL_worst
    df['PF_avg'] = PF_avg
    df['PF_worst'] = PF_worst
    df.plot()
    plt.xlabel('Epoch Number')
    plt.ylabel('User Data Rate')
    plt.show()




