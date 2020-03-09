import tensorflow as tf
import pandas as pd
import numpy as np
import os

class Network:
    def __init__(self):
        pass

    def conv_block(self, input_tensor, num_filters, kernel_size=(3,3), strides=(1,1), padding='same'):
        x = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return x

    def res_block(self, input_tensor, num_filters, kernel_size=(3,3), strides=(1,1), padding='same'):
        filters1, filters2, filters3 = num_filters
        x = self.conv_block(input_tensor,filters1, kernel_size, strides, padding)
        x = self.conv_block(x, filters2, kernel_size, strides, padding)
        x = self.conv_block(x, filters3, kernel_size, strides, padding)

        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build(self, inputs):
        x = self.conv_block(inputs, 64)

        ''' Downsampling 1 '''
        x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 3), padding='same', name='DownSampling-1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = self.res_block(x, (64, 64, 128))

        ''' Downsampling 2 '''
        x = tf.keras.layers.Conv2D(256, kernel_size=(1, 2), strides=(1, 2), padding='valid', name='DownSampling-2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = self.res_block(x, (128, 128, 256))

        ''' Downsampling 3 '''
        x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 3), padding='same', name='DownSampling-3')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return x

class Actor:
    def __init__(self, user_num, feature_num, rbg_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rbg_num = rbg_num
        self.lr = lr

    def get_network(self, inputs):
        x = Network().build(inputs)

        x = tf.keras.layers.Conv2D(self.rbg_num, (1, 1))(x)
        outputs = tf.keras.layers.Softmax(axis=1)(x[:, :, 0, :])

        return outputs

    def build(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num, 1])
        outputs = self.get_network(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )

        return model

class Critic:
    def __init__(self, user_num, feature_num, rbg_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rbg_num = rbg_num
        self.lr = lr

    def get_network(self, inputs):
        x = Network().build(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1)(x)

        return outputs

    def build(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num + self.rbg_num, 1])

        outputs = self.get_network(inputs)

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
            columns=['state_user_info', 'state_rbg_avl', 'state_tx_user' ,
                     'action', 'reward',
                     'next_state_user_info', 'next_state_rbg_avl', 'next_state_tx_user',
                     'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, frac):
        self.memory.dropna(inplace=True)
        sample = self.memory.sample(frac=frac)
        sample = sample[sample['state_tx_user'] == sample['next_state_tx_user']]

        sample_shape_list = []

        for index in sample.index:
            sample_shape_list.append(len(self.memory.loc[index]['state_tx_user']))
        sample_shape_list = list(set(sample_shape_list) & set(sample_shape_list))

        sub_sample_list = []
        for i in range(len(sample_shape_list)):
            sub_sample_list.append([])
            for index in sample.index:
                if len(self.memory.loc[index]['state_tx_user']) == sample_shape_list[i]:
                    sub_sample_list[i].append(index)

        return sub_sample_list

    def clear(self):
        self.__init__(capacity=self.capacity)

class DDPGAgent:
    def __init__(self,
                 user_num,
                 feature_num,
                 rbg_num,
                 gamma=0.99,
                 sample_frac=0.1,
                 lr=0.001,
                 replayer_capacity=10000,
                 save_dir='./snapshots/',
                 log_dir='./log',
                 resume=False,
                 checkpoint=None):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rbg_num = rbg_num

        self.gamma = gamma
        self.lr = lr
        self.sample_frac = sample_frac
        self.replayer = Replayer(replayer_capacity)
        self.save_dir = save_dir
        self.log_dir = os.path.join(log_dir)
        self.resume = resume
        self.checkpoint = checkpoint

        self.actor = Actor(
            self.user_num,
            self.feature_num,
            self.rbg_num, self.lr)
        self.critic = Critic(
            self.user_num,
            self.feature_num,
            self.rbg_num, self.lr)

        if resume:
            self.actor_eval_net = tf.keras.models.load_model(self.checkpoint['actor_eval'])
            self.actor_target_net = tf.keras.models.load_model(self.checkpoint['actor_eval'])
            self.critic_eval_net = tf.keras.models.load_model(self.checkpoint['critic_eval'])
            self.critic_target_net = tf.keras.models.load_model(self.checkpoint['critic_eval'])
        else:
            self.actor_eval_net = self.actor.build()
            self.actor_target_net = self.actor.build()
            self.critic_eval_net = self.critic.build()
            self.critic_target_net = self.critic.build()

        self.update_target_net(self.actor_target_net, self.actor_eval_net, self.lr)
        self.update_target_net(self.critic_target_net, self.critic_eval_net, self.lr)

    def update_target_net(self, target_net, eval_net, lr):
        target_weights = target_net.get_weights()
        eval_weights = eval_net.get_weights()

        avg_weights = [(1. - lr) * t + lr * e
                       for t, e in zip(target_weights, eval_weights)]

        target_net.set_weights(avg_weights)

    def decide(self, state):
        ''' no user, no action '''
        if len(state['tx_user']) == 0:
            return {'action': None, 'softmax_score': None}
        else:
            ob = state['user_info']
            ob = tf.convert_to_tensor(ob, tf.float32)
            ob = tf.expand_dims(ob, 0)

            pred = self.actor_eval_net.predict(ob[..., tf.newaxis])
            action = tf.argmax(pred, axis=1).numpy().tolist()[0]

            for i in range(self.rbg_num):
                if state['rbg_avl'][i] is not None:
                    action[i] = None
                    continue
                for j in range(len(state['tx_user'])):
                    if action[i] == j:
                        action[i] = state['tx_user'][j]

            return {'action': action, 'softmax_score': pred[0]}

    def learn(self, state, action, reward, next_state, done, store=True):
        if store and action is not None:
            self.replayer.store(state['user_info'], state['rbg_avl'], state['tx_user'],
                                action, reward,
                                next_state['user_info'], next_state['rbg_avl'], next_state['tx_user'],
                                done)

        if done:
            sample = self.replayer.sample(self.sample_frac)
            for sub_sample in sample:
                batch_sample = (np.stack(self.replayer.memory.loc[sub_sample, field], axis=0) for field in self.replayer.memory.columns)
                state_user_info, state_rbg_avl, state_tx_user, \
                action_, reward_, \
                next_state_user_info, next_state_rbg_avl, next_state_tx_user, \
                done_ = batch_sample

                state_ = state_user_info
                next_state_ = next_state_user_info

                state_tensor = tf.convert_to_tensor(state_, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    action_tensor = self.actor_eval_net(state_tensor[..., tf.newaxis])
                    # build the input, shape=[1, user_num, feature_num + rbg_num, 1]
                    input_tensor = tf.concat([state_tensor[..., tf.newaxis], action_tensor[..., tf.newaxis]], axis=2)
                    q_tensor = self.critic_eval_net(input_tensor)
                    loss_tensor = - tf.reduce_mean(q_tensor)
                    grad_tensors = tape.gradient(loss_tensor, self.actor_eval_net.variables)
                    self.actor_eval_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_eval_net.variables))
                    tf.print("actor loss: ", loss_tensor)

                next_actions = self.actor_target_net.predict(next_state_[..., tf.newaxis])
                # state_actions.shape = [self.batchs, user_num, feature_num + rbg_num]
                state_actions = tf.concat([state_, action_], axis=2)
                # nex_state_actions.shape = [self.batchs, user_num, feature_num + rbg_num]
                next_state_actions = tf.concat([next_state_, next_actions], axis=2)
                next_qs = self.critic_target_net.predict(next_state_actions[..., tf.newaxis])[:, 0]
                targets = reward_ + self.gamma * next_qs * (1. - done_)
                # print('targets={}'.format(targets))

                self.critic_eval_net.fit(state_actions[..., tf.newaxis], targets)

                self.update_target_net(self.actor_target_net,
                                       self.actor_eval_net, self.lr)
                self.update_target_net(self.critic_target_net,
                                       self.critic_eval_net, self.lr)


    def save(self, epoch):
        self.actor_eval_net.save(self.save_dir + 'actor_eval_net_epoch' + str(epoch) + '.h5')
        self.critic_eval_net.save(self.save_dir + 'critic_eval_net_epoch' + str(epoch) + '.h5')


# actor = Actor(None, feature_num=21, rbg_num=17, lr=0.001)
# net = actor.build()
# net.summary()
# user_sum = len(list(range(2)))
# to_newtx_ues_idx = [1, 3, 7]
# newtx_rbg_ue = [0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# user_info = np.random.rand(user_sum, 21)
#
# inputs = np.expand_dims(user_info, 0)
# pred = net.predict(inputs[..., np.newaxis])
# print(pred)
# pred = list(tf.argmax(pred, axis=0).numpy())
# print(pred)
#
# print(pred)
# for i in range(17):
#     if newtx_rbg_ue[i] == None:
#         pred[i] = None
#         continue
#     for j in range(len(to_newtx_ues_idx)):
#         if pred[i] == j:
#             pred[i] = to_newtx_ues_idx[j]
#
# print(pred)
