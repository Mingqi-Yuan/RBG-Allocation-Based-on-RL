import tensorflow as tf
import pandas as pd
import numpy as np
import os

class Network:
    def __init__(self):
        pass

    def identity_block(self, input_tensor, num_filters, kernel_size, strides=(1, 1), padding='same'):
        filters1, filters2, filters3 = num_filters
        x = tf.keras.layers.Conv2D(filters1, kernel_size, strides, padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(filters2, kernel_size, strides, padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(filters3, kernel_size, strides, padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def conv_block(self, input_tensor, num_filters, kernel_size, strides=(1, 1), padding='same'):
        filters1, filters2, filters3 = num_filters
        x = tf.keras.layers.Conv2D(filters1, kernel_size, strides, padding)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(filters2, kernel_size, strides, padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(filters3, kernel_size, strides, padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def build(self, inputs):
        x = self.conv_block(inputs, num_filters=[64, 64, 256], kernel_size=[3,3])
        # x = self.identity_block(x, num_filters=[64, 64, 256], kernel_size=[3,3])
        # x = self.identity_block(x, num_filters=[64, 64, 256], kernel_size=[3,3])

        # x = self.conv_block(x, num_filters=[128, 128, 512], kernel_size=[3,3])
        # x = self.identity_block(x, num_filters=[128, 128, 512], kernel_size=[3,3])
        # x = self.identity_block(x, num_filters=[128, 128, 512], kernel_size=[3,3])

        return x

class Actor:
    def __init__(self, user_num, feature_num, rgb_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num
        self.lr = lr

    def get_network(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num, 1])

        backbone = Network().build(inputs)

        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(backbone)

        part1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 7), strides=(1, 7), padding='same')(x)
        part2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 3), padding='same')(x)
        part3 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 3), padding='same')(x)

        x = tf.keras.layers.concatenate([part1, part2, part3], axis=2)
        x = tf.keras.backend.mean(x, axis=3)
        outputs = tf.math.softmax(x, axis=1)

        model = tf.keras.Model(inputs=inputs, outputs = outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )

        return model

class Critic:
    def __init__(self, user_num, feature_num, rgb_num, lr):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num
        self.lr = lr

    def get_network(self):
        inputs = tf.keras.Input([self.user_num, self.feature_num + self.rgb_num, 1])

        x = Network().build(inputs)

        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
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
                 log_dir='./log',
                 resume=False,
                 checkpoint=None):
        self.user_num = user_num
        self.feature_num = feature_num
        self.rgb_num = rgb_num

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replayer = Replayer(replayer_capacity)
        self.save_dir = save_dir
        self.log_dir = os.path.join(log_dir)
        self.resume = resume
        self.checkpoint = checkpoint

        self.actor = Actor(
            self.user_num,
            self.feature_num,
            self.rgb_num, self.lr)
        self.critic = Critic(
            self.user_num,
            self.feature_num,
            self.rgb_num, self.lr)

        if resume:
            self.actor_eval_net = tf.keras.models.load_model(self.checkpoint['actor_eval'])
            self.actor_target_net = tf.keras.models.load_model(self.checkpoint['actor_target'])
            self.critic_eval_net = tf.keras.models.load_model(self.checkpoint['critic_eval'])
            self.critic_target_net = tf.keras.models.load_model(self.checkpoint['critic_target'])

        self.actor_eval_net = self.actor.get_network()
        self.actor_target_net = self.actor.get_network()
        self.critic_eval_net = self.critic.get_network()
        self.critic_target_net = self.critic.get_network()

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
        pred = self.actor_eval_net.predict(ob[..., tf.newaxis])

        action = tf.argmax(pred, axis=1)[0].numpy()

        return {'action':action, 'softmax_score':pred[0]}

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)

        if done:
            observation_, action_, reward_, next_observation_, done_ = self.replayer.sample(self.batch_size)

            observation_tensor = tf.convert_to_tensor(observation_, dtype=tf.float32)
            observation_tensor = tf.expand_dims(observation_tensor, 3)

            with tf.GradientTape() as tape:
                action_tensor = self.actor_eval_net(observation_tensor)
                # build the input, shape=[1, user_num, feature_num + rgb_num, 1]
                input_tensor = tf.concat([observation_tensor, action_tensor[..., tf.newaxis]], axis=2)
                q_tensor = self.critic_eval_net(input_tensor)
                loss_tensor = - tf.reduce_mean(q_tensor)
                grad_tensors = tape.gradient(loss_tensor, self.actor_eval_net.variables)
                self.actor_eval_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_eval_net.variables))

            next_actions = self.actor_target_net.predict(next_observation_[..., tf.newaxis])
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

