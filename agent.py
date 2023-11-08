"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import Adam


class NeuralNetwork(nn.Module):
    def __init__(self, layers: nn.Sequential, learning_rate, loss, device):
        super(NeuralNetwork, self).__init__()
        self.layers = layers
        self.optimizer = torch.optim.RMSprop(self.parameters(), learning_rate)
        self.loss_function = loss
        self.device = device

    def forward(self, x: np.ndarray):
        tensor = torch.from_numpy(x).to(device=self.device)
        return self.layers(tensor)

    def get_weights(self):
        return deepcopy(self._parameters)

    def set_weights(self, parameters):
        self._parameters = parameters
        return

    def print_summary(self):
        print("+------------------- Summary ------------------+")
        print("|         Layer         |        Shape         |")
        print("+-----------------------+----------------------+")
        for name, parameter in self.named_parameters():
            print(f"| {name:<21} | {str(list(parameter.shape)):<20} |")
        print("+----------------------------------------------+\n")

    def train_on_batch(self, states, targets):
        targets = torch.tensor(targets, device=self.device, dtype=torch.float32)
        self.train()
        output = self(states)
        loss = self.loss_function(output, targets)

        # clear gradients for this training step
        self.optimizer.zero_grad()

        # backpropagation, compute gradients
        loss.backward()  # apply gradients
        self.optimizer.step()
        return loss.cpu().detach().numpy()


class Agent:
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """

    def __init__(
        self,
        board_size=10,
        frames=2,
        buffer_size=10000,
        gamma=0.99,
        n_actions=3,
        use_target_net=True,
        version="",
    ):
        """initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._n_frames, self._board_size, self._board_size)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(
            self._board_size, -1
        )
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer

        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(
            self._buffer_size, self._board_size, self._n_frames, self._n_actions
        )

    def get_buffer_size(self):
        """Get the current buffer size

        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path="", iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), "wb") as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path="", iteration=None):
        """Load the buffer from disk

        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), "rb") as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point // self._board_size, point % self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row * self._board_size + col


class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model
    """

    def __init__(
        self,
        board_size=10,
        frames=4,
        buffer_size=10000,
        gamma=0.99,
        n_actions=3,
        use_target_net=True,
        version="",
    ):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(
            self,
            board_size=board_size,
            frames=frames,
            buffer_size=buffer_size,
            gamma=gamma,
            n_actions=n_actions,
            use_target_net=use_target_net,
            version=version,
        )
        self.reset_models()

    def reset_models(self):
        """Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if self._use_target_net:
            self._target_net = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board: np.ndarray):
        """Reshape input and normalize

        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : TensorFlow Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board,
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model.eval()
        with torch.no_grad():
            model_outputs = model(board)
        return model_outputs.cpu().numpy()

    def _normalize_board(self, board):
        """Normalize the board before input to the network

        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value

        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def _agent_model(self) -> NeuralNetwork:
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph
        """
        # define the input layer, shape is dependent on the board size and frames
        with open("model_config/{:s}.json".format(self._version), "r") as f:
            m = json.loads(f.read())

        layers = []
        calculation_sample = torch.from_numpy(
            np.zeros((64,) + self._input_shape, dtype=np.float32)
        )
        for name in m["model"]:
            layer = None
            l = m["model"][name]
            if "Conv2D" in name:
                # add convolutional layer
                layer = nn.Conv2d(
                    in_channels=calculation_sample.shape[1],
                    out_channels=l["filters"],
                    kernel_size=l["kernel_size"],
                    padding="same",
                )
            elif "Flatten" in name:
                layer = nn.Flatten()
            elif "Dense" in name:
                layer = nn.Linear(
                    out_features=l["units"], in_features=calculation_sample.shape[1]
                )
            elif "activation" in l:
                if l["activation"] == "relu":
                    layer = nn.ReLU()
                else:
                    raise Exception(
                        "activation function {:s} not supported".format(l["activation"])
                    )
            else:
                raise Exception("Layer type not supported".format(name))
            layers.append(layer)
            calculation_sample = layer(calculation_sample)

        layers.append(
            nn.Linear(
                out_features=self._n_actions, in_features=calculation_sample.shape[1]
            )
        )

        return NeuralNetwork(
            nn.Sequential(*layers),
            learning_rate=0.0005,
            loss=nn.CrossEntropyLoss(),
            device="mps",
        ).to(device="mps")

    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ["action_prev_dense", "action_values"]:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer=self._model.optimizer, loss=self._model.loss)

    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs

    def save_model(self, file_path="", iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)

        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(
            self._model.state_dict(), "{}/model_{:04d}.h5".format(file_path, iteration)
        )
        if self._use_target_net:
            torch.save(
                self._target_net.state_dict(),
                "{}/model_{:04d}_target.h5".format(file_path, iteration),
            )

    def load_model(self, file_path="", iteration=None):
        """load any existing models, if available"""
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(
            torch.load("{}/model_{:04d}.h5".format(file_path, iteration))
        )
        if self._use_target_net:
            self._target_net.load_state_dict(
                torch.load("{}/model_{:04d}_target.h5".format(file_path, iteration))
            )
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print("Training Model")
        self._model.print_summary()
        if self._use_target_net:
            print("Target Network")
            self._target_net.print_summary()

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward +
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions

        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = r + (
            self._gamma
            * np.max(
                np.where(legal_moves == 1, next_model_outputs, -np.inf),
                axis=1,
            ).reshape(-1, 1)
        ) * (1 - done)

        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s)
        # we bother only with the difference in reward estimate at the selected action
        target = (1 - a) * target + a * discounted_reward
        # fit
        loss = self._model.train_on_batch(self._normalize_board(s), target)
        # loss = round(loss, 5)
        return loss

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if self._use_target_net:
            self._target_net.set_weights(self._model.get_weights())

    def compare_weights(self):
        """Simple utility function to heck if the model and target
        network have the same weights or not
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (
                    self._model.layers[i].weights[j].numpy()
                    == self._target_net.layers[i].weights[j].numpy()
                ).all()
                print("Layer {:d} Weights {:d} Match : {:d}".format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())
