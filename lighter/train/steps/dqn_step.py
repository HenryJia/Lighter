import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from apex import amp

from .steps import StepReport, RLStep



class DQNStep(RLStep):
    """
    The default supervised step class that runs basic supervised training

    This is pretty much the most basic possible step. It just handles runnign the model and applying the loss and metrics

    Returns a StepReport containing the outputs of the model, the loss and the metrics

    Parameters
    ----------
    env: Training environment
        This is the environment we will sample from to train
        Ideally, this should be an OpenAI Gym environment but anything which has the same API works
    model: PyTorch model
        The PyTorch model we want to optimize
    policy: Policy class
        Class for policy to apply
    memory: Memory class
        Class for replay memory
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, agent, policy, memory, optimizer, batch_size, metrics=[], train=True, use_amp=False):
        self.env = env
        self.agent = agent
        self.policy = policy
        self.memory = memory
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp


    def episode_begin(self): # Reset our environment back to the beginning
        self.state = self.env.reset()


    def __call__(self): # We don't actually need any data since the environment is a member variable
        with torch.no_grad():
            device = next(self.agent.parameters()).device

            state = torch.tensor(self.state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            out = self.agent(state.view(1, -1))
            action = self.policy(out)

            next_state, reward, done, info = self.env.step(action)
            self.state = next_state if not done else None

            metrics = [('reward', reward)]

        if self.train:
            action = torch.tensor(action).to(device=device, dtype=torch.long, non_blocking=True)
            reward = torch.tensor(reward).to(device=device, dtype=torch.float32, non_blocking=True)

            if done:
                self.memory.push([state, action, reward, None])
            else:
                next_state = torch.from_numpy(next_state).to(device=device, dtype=torch.float32, non_blocking=True)
                self.memory.push([state, action, reward, next_state])

            if len(self.memory) > self.batch_size:
                self.optimizer.zero_grad()
                loss = self.agent.backward(self.memory.sample(self.batch_size))

                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                for param in self.agent.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                with torch.no_grad():
                    # Compute the metrics
                    metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

                return StepReport(outputs = {'out': out, 'action': action}, losses={'loss': loss.item()}, metrics=dict(metrics)), done

        return StepReport(outputs = {'out': out, 'action': action}, losses={'loss': 0}, metrics=dict(metrics)), done

