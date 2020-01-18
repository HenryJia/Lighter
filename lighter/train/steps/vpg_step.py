import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.optim import Adam

from apex import amp

from .steps import StepReport



class VPGStep(object):
    """
    The Monte Carlo vanilla policy gradient step class

    Returns a StepReport containing the outputs of the model, the loss and the metrics

    Parameters
    ----------
    env: Training environment
        This is the environment we will sample from to train
        Ideally, this should be an OpenAI Gym environment but anything which has the same API works
    model: PyTorch model
        The PyTorch model we want to optimize
        Note the output should be a PyTorch distribtion not a tensor so we can sample and comptue log probs easily
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    gamma: Float
        Discount factor for computing the loss, default is 0.9
    epsilon:
        Small epsilon fuzz parameter to be used for certain ops
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
    train: Boolean
        Whether we are training
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, agent, optimizer, gamma=0.9, epsilon=1e-5, metrics=[], train=True, use_amp=False):
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp


    def reset(self): # Reset our environment back to the beginning
        self.state = self.env.reset()

        # Note: We only store history for the episode, so assume we can store the entire episode in GPU memory
        self.log_probs = []
        self.reward_history = []

    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.agent.parameters()).device

        state = torch.tensor(self.state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        out_distribtion = self.agent(state.view(1, -1))
        action = out_distribtion.sample()

        next_state, reward, done, info = self.env.step(action.item())
        self.state = next_state if not done else None

        metrics = [('reward', reward)]

        if self.train:
            self.log_probs.append(out_distribtion.log_prob(action))

            #reward = torch.tensor(reward).to(device=device, dtype=torch.float32, non_blocking=True)
            # Keep rewards out of PyTorch, we have to loop through and standard python should be a little faster
            self.reward_history.append(reward)

            if done:
                returns = []
                R = 0
                # Compute the value at each time step with discount factor
                # Fastest way  to do this is to loop backwards through the rewards
                for r in self.reward_history[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)

                returns = torch.tensor(returns).to(device=device, dtype=torch.float32, non_blocking=True)
                returns = (returns - returns.mean()) / (returns.std() + self.epsilon)

                self.log_probs = torch.cat(self.log_probs, dim=0)
                loss = -torch.sum(self.log_probs * returns)

                self.optimizer.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                #with torch.no_grad():
                    ## Compute the metrics, this does nothing currently
                    #metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

                return StepReport(outputs = {'out': out_distribtion, 'action': action}, losses={'loss': loss.item()}, metrics=dict(metrics)), done

        return StepReport(outputs = {'out': out_distribtion, 'action': action}, losses={'loss': 0}, metrics=dict(metrics)), done

