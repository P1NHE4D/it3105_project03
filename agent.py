import numpy as np
from keras.layers import Dense
from keras.models import Model, Sequential

from interface import Domain
import tensorflow as tf
from tqdm import tqdm


class Agent:

    def __init__(
            self,
            domain: Domain,
            learning_rate=0.01,
            epsilon=0,
            epsilon_decay=0.99,
            episodes=20,
            steps=300,
            gamma=0.99,
            hidden_layers=None,
            weight_file="",
            filepath="models"
    ):
        self.domain = domain
        state, action = domain.get_init_state()
        input_shape = (state.shape[0] + action.shape[1],)
        if hidden_layers is None:
            hidden_layers = ["relu", 32]
        self.qnet = QNET(
            input_shape=input_shape,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            weight_file=weight_file
        )
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.steps = steps
        self.gamma = gamma
        self.filepath = filepath

    def train(self):

        progress = tqdm(range(1, self.episodes + 1))

        for _ in progress:
            state, actions = self.domain.get_init_state()
            action = self.propose_action(state=state, actions=actions, epsilon=self.epsilon)

            for step in range(self.steps):
                successor_state, actions, reward = self.domain.get_child_state(action)
                x = np.concatenate([state, action])
                y = reward
                if self.domain.is_current_state_terminal():
                    self.qnet.fit(x=np.array([x]), y=np.array([y]), verbose=3)
                    break
                successor_action, sa_value = self.propose_action(
                    state=successor_state,
                    actions=actions,
                    epsilon=self.epsilon,
                    return_sa_value=True
                )
                y += self.gamma * sa_value
                self.qnet.fit(x=np.array([x]), y=np.array([y]), verbose=3)
                state = successor_state
                action = successor_action


                progress.set_description(
                    "Epsilon: {}".format(self.epsilon) +
                    " | Step: {}/{}".format(step, self.steps)
                )
            self.epsilon *= self.epsilon_decay

        # store learned weights
        self.qnet.save_weights(filepath=self.filepath)

    def propose_action(self, state, actions, epsilon=0, return_sa_value=False):
        # random action with probability epsilon
        if np.random.random() < epsilon:
            action_idx = np.random.choice(actions.shape[0])
            action = actions[action_idx]
            if return_sa_value:
                sa_value = self.qnet.predict(np.array([np.concatenate([state, action])]))[0][0]
                return action, sa_value
            return action

        # best action based on state-action evaluations
        sa_values = [self.qnet.predict(np.array([np.concatenate([state, action])]))[0][0] for action in actions]
        idx = np.argmax(sa_values)
        if return_sa_value:
            return actions[idx], sa_values[idx]
        return actions[idx]


class QNET(Model):

    def __init__(self, input_shape, learning_rate, hidden_layers, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_layers = [Dense(nodes, activation if activation != "linear" else None) for nodes, activation in
                         hidden_layers]
        hidden_layers.append(Dense(1))
        self.model = Sequential(layers=hidden_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.mse
        )

        self.model_trained = False
        try:
            self.load_weights(weight_file)
            self.model_trained = True
        except Exception as e:
            print("Unable to load weight file", e)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super().get_config()
