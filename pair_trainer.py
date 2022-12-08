import torch
from transformers import Trainer

class PairwiseTrainer(Trainer):
    """To be used with models that return only a single value per input."""
    def compute_loss(self, model, inputs, return_outputs=False):
        bs = inputs["input_ids"].shape[0] // 2
        # forward pass
        rewards = model(**inputs)
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, rewards) if return_outputs else loss


PAD_ID=50256
max_len=1024
class PairwiseDivergenceTrainer(Trainer):
    """To be used with models that return a value for each token in the input."""
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        chosen = inputs["input_ids"][:bs]
        rejected = inputs["input_ids"][bs:]
        rewards = model(**inputs)
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss. Only backprop on last value before padding
        loss = 0
        for i in range(bs):
            # Retrieve first index where trajectories diverge
            divergence = (chosen[i] != rejected[i]).nonzero()
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0] if len(divergence) > 0 else max_len
            assert divergence_ind > 0
            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)
            # Index into correct reward
            c_truncated_reward = chosen_rewards[i][divergence_ind : end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind : end_ind]
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs
        return (loss, rewards) if return_outputs else loss

# calculate the truncated reward as above
def truncated_reward(input_ids, rewards):
    bs = input_ids.size(0)
    for i in range(bs):
        # Retrieve first index where trajectories diverge
        # Check if there is any padding otherwise take length of sequence
        inds = (chosen[i] == PAD_ID).nonzero()
        c_ind = inds[0].item() if len(inds) > 0 else chosen.shape[1]
        # Index into correct reward
        c_truncated_reward = chosen_rewards[i][divergence_ind : c_ind]
        return c_truncated_reward