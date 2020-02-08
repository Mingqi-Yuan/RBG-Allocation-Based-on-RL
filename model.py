import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os

from simulator import AIRVIEW
from simulator import load_av_ue_info

class Actor:
    def __init__(self, user_num, feature_num, rgb_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num
        self.lr = lr

    def network(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num, 1])

        x = tf.keras.layers.Conv2D(128, (1, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        x = tf.keras.layers.Conv2D(256, (1, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        x = tf.keras.layers.Conv2D(516, (1, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        outputs = tf.keras.layers.Conv2D(1024, (1, 3), padding='same')(x)

        model = tf.keras.Model(inputs=inputs, outputs = outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )

        return model

    def actor_head(self, outputs):
        part1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 7), strides=(1, 7))(outputs)
        part2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(outputs)
        part3 = tf.keras.layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 3))(outputs)

        pred = tf.keras.layers.concatenate([part1, part2, part3], axis=2)
        pred = tf.keras.backend.mean(pred, axis=3)
        pred = tf.math.softmax(pred, axis=1)

        return pred

class Critic:
    def __init__(self, user_num, feature_num, rgb_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num
        self.lr = lr

    def network(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num + self.rgb_num, 1])

        x = tf.keras.layers.Conv2D(128, (1, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        x = tf.keras.layers.Conv2D(256, (1, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        x = tf.keras.layers.Conv2D(516, (1, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.01)(x)

        x = tf.keras.layers.Conv2D(1024, (1, 3), padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )

        return model

class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['observation', 'action',
                     'reward', 'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)
    def clear(self):
        self.__init__(capacity=self.capacity)

class DDPGAgent:
    def __init__(self,
                 user_num,
                 feature_num,
                 rgb_num,
                 gamma=0.99,
                 batch_size = 32,
                 lr=0.001,
                 replayer_capacity=100000,
                 save_dir='./snapshots/',
                 log_dir='./log'):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replayer = Replayer(replayer_capacity)
        self.save_dir = save_dir
        self.log_dir = os.path.join(log_dir)

        self.actor = Actor(
            self.user_num,
            self.feature_num,
            self.rgb_num, self.lr)
        self.critic = Critic(
            self.user_num,
            self.feature_num,
            self.rgb_num, self.lr)
        self.actor_eval_net = self.actor.network()
        self.actor_target_net = self.actor.network()
        self.critic_eval_net = self.critic.network()
        self.critic_target_net = self.critic.network()

        self.update_target_net(self.actor_target_net, self.actor_eval_net, self.lr)
        self.update_target_net(self.critic_target_net, self.critic_eval_net, self.lr)

    def update_target_net(self, target_net, eval_net, lr):
        target_weights = target_net.get_weights()
        eval_weights = eval_net.get_weights()

        avg_weights = [(1. - lr) * t + lr * e
                       for t,e in zip(target_weights, eval_weights)]

        target_net.set_weights(avg_weights)

    def decide(self, observation):
        ob = tf.convert_to_tensor(observation, tf.float32)
        ob = tf.expand_dims(ob, 0)
        output = self.actor_eval_net.predict(ob[..., tf.newaxis])
        pred = self.actor.actor_head(output)

        action = tf.argmax(pred, axis=1)[0].numpy()

        return {'action':action, 'softmax_score':pred[0]}

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)

        observation_, action_, reward_, next_observation_, done_ = self.replayer.sample(self.batch_size)

        observation_tensor = tf.convert_to_tensor(observation_, dtype=tf.float32)
        observation_tensor = tf.expand_dims(observation_tensor, 3)

        with tf.GradientTape() as tape:
            action_tensor = self.actor_eval_net(observation_tensor)
            action_tensor = self.actor.actor_head(action_tensor)
            # build the input, shape=[1, user_num, feature_num + rgb_num, 1]
            input_tensor = tf.concat([observation_tensor, action_tensor[..., tf.newaxis]], axis=2)
            q_tensor = self.critic_eval_net(input_tensor)
            loss_tensor = - tf.reduce_mean(q_tensor)
            grad_tensors = tape.gradient(loss_tensor, self.actor_eval_net.variables)
            self.actor_eval_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_eval_net.variables))

        next_actions = self.actor_target_net.predict(next_observation_[..., tf.newaxis])
        next_actions = self.actor.actor_head(next_actions)
        # observation_actions.shape = [self.batchs, user_num, feature_num + rgb_num]
        observation_actions = tf.concat([observation_, action_], axis=2)
        # nex_observation_actions.shape = [self.batchs, user_num, feature_num + rgb_num]
        next_observation_actions = tf.concat([next_observation_, next_actions], axis=2)
        next_qs = self.critic_target_net.predict(next_observation_actions[..., tf.newaxis])[:, 0]
        targets = reward_ + self.gamma * next_qs * (1. - done_)

        self.critic_eval_net.fit(observation_actions[..., tf.newaxis], targets)

        self.update_target_net(self.actor_target_net,
                               self.actor_eval_net, self.lr)
        self.update_target_net(self.critic_target_net,
                               self.critic_eval_net, self.lr)

    def save(self, epoch):
        self.actor_eval_net.save(self.save_dir + 'actor_eval_net_epoch' + str(epoch) + '.h5')
        self.critic_eval_net.save(self.save_dir + 'critic_eval_net_epoch' + str(epoch) + '.h5')
        self.actor_target_net.save(self.save_dir + 'actor_target_net_epoch' + str(epoch) + '.h5')
        self.critic_target_net.save(self.save_dir + 'critic_target_net_epoch' + str(epoch) + '.h5')

if __name__== '__main__':
    BATCH_SIZE = 100
    EPOCH = 1500
    MEMORY_CAPACITY = 1000000
    RGB_NUM = 17
    USER_NUM = None
    FEATURE_NUM = 21

    env = AIRVIEW()
    av_ues_info = load_av_ue_info()
    agent = DDPGAgent(
        user_num=USER_NUM,
        feature_num=FEATURE_NUM,
        rgb_num=RGB_NUM,
        batch_size=BATCH_SIZE,
        replayer_capacity=MEMORY_CAPACITY)
    performance = dict()

    for i in range(EPOCH):
        print('Epoch: {}'.format(i))
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
            # if env.bs.newtx_rbg_ue == [None for _ in range(RGB_NUM)]:
            #     observation_next, reward, done, info = env.step(None)
            #     observation_next = tf.convert_to_tensor(observation_next, dtype=tf.float32)
            #     observation_next = tf.expand_dims(observation_next, 0)
            # else:
            action = agent.decide(observation)
            print('Allocation result: ',action['action'])
            next_observation, reward, done, _ = env.step(action['action'])
            agent.learn(observation, action['softmax_score'], reward, next_observation, done)
            if done:
                rl_result = env.get_result()
                performance['RL'] = performance.get('RL', [])
                performance['RL'].append((rl_result['avg_ue_rate'], rl_result['worst_ue_rate']))
                break
            observation = next_observation

        # clear the replayer and save the model
        agent.replayer.clear()
        agent.save(epoch=i)

        with open('record.txt', 'a') as file:
            record = '{} Epoch={}, ue_idx={}~{}, {} ues_rate:[avg={},worst={}], {} ues_rate:[avg={},worst={}]'.format(
                datetime.datetime.now(), i, env.av_ues_idx[0], env.av_ues_idx[-1],
                'PF', pf_result['avg_ue_rate'], pf_result['worst_ue_rate'],
                'RL', rl_result['avg_ue_rate'], rl_result['worst_ue_rate'])
            file.write(record)
            file.write('\n')
            file.close()

        if i % 10 == 0 and i is not 0:
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



