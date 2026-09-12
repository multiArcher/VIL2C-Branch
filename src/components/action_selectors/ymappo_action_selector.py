from torch.distributions import Categorical

from .action_selector import ActionSelector


class YMAPPOSoftPoliciesSelector(ActionSelector):
    """Sample from the policy in training; greedy argmax when test_greedy is set."""

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked = agent_inputs.clone()
        masked[avail_actions == 0] = 0.0
        if test_mode and getattr(self.args, "test_greedy", True):
            return masked.max(dim=-1)[1].long()
        dist = Categorical(masked)
        return dist.sample().long()
