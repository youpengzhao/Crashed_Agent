import copy
import os
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop



class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def test(self, batch: EpisodeBatch):    # This function is used to test the Q_total value of the models.
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if self.args.offline_path != "":
            timesteps = []
            timestep_to_load = 0

        # Go through all files in args.offline_path
            for name in os.listdir(self.args.offline_path):
                full_name = os.path.join(self.args.offline_path, name)
            # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))
            if self.args.load_step == 0:
            # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
            # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))
            model_path = os.path.join(self.args.offline_path, str(timestep_to_load))
            self.target_mac.init_hidden(batch.batch_size)
            self.target_mac.load_models(model_path)
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(model_path), map_location=lambda storage, loc: storage))
        
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        if self.args.offline:   # In crashed environments, mac_out restore the output of the model in offline_path.
            for t in range(batch.max_seq_length):
                agent_outs = self.target_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        else:
            for t in range(batch.max_seq_length):    # If the environment is normal, the output is generated by the model in checkpoint_path.
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        f1 = './Q_act.txt'          # Record the Q value of every agent in this file.
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        print(chosen_action_qvals.shape)
        with open(f1, "w") as file:
            for i in range(self.args.n_agents):
                file.write("agent: "+ str(i) + str(chosen_action_qvals[:, :, i].tolist()))
                file.write('\n')
        # the last dim---the Q of every agent
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        for t in range(batch.max_seq_length):
            target_agent_outs = self.mac.forward(batch, t=t)    # theory value
            target_mac_out.append(target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximize live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        # Mix
        if self.mixer is not None:
            if self.args.offline:
                chosen_action_qvals, reg_loss = self.target_mixer(chosen_action_qvals, batch["state"][:, :-1])   # offline value
            else:
                chosen_action_qvals, reg_loss = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals, _ = self.mixer(target_max_qvals, batch["state"][:, 1:])   # theory
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        n = len(rewards[0])
        res = copy.deepcopy(rewards)
        for i in range(n-2, -1, -1):
            res[0][i] = rewards[0][i] + self.args.gamma*res[0][i+1]
       
        f2 = './Q_total.txt'      # Record the target Q_value, chosen_action_qvals_total, cumulative rewards and the rewards in this file.
        with open(f2, "w") as file:
            file.write("target_max_qvals: " + str(targets.tolist()))
            file.write('\n')
            file.write("chosen_action_qvals_total " + str(chosen_action_qvals.tolist()))
            file.write('\n')
            file.write("calculate rewards " + str(res.tolist()))
            file.write('\n')
            file.write("rewards " + str(rewards.tolist()))
        
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        mask_elems = mask.sum().item()
        print((chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents))
        print((targets * mask).sum().item()/(mask_elems * self.args.n_agents))

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
