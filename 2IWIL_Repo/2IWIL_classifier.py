import argparse
from itertools import count

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from loss import *

"""
2IWIL: proposed method (--weight)
GAIL (U+C): no need to specify option
GAIL (Reweight): use only confidence data (--weight --only)
GAIL (C): use only confidence data without reweighting (--weight --only --noconf)
"""

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed (default: 1111')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='size of a single batch')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='interval between training model save (default: 100)')
parser.add_argument('--fname', type=str, default='expert', metavar='F',
                    help='the file name to save trajectory')
parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train an expert')
parser.add_argument('--hidden-dim', type=int, default=100, metavar='H',
                    help='the size of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='learning rate')
parser.add_argument('--weight', action='store_true',
                    help='consider confidence into loss')
parser.add_argument('--only', action='store_true',
                    help='only use labeled samples')
parser.add_argument('--noconf', action='store_true',
                    help='use only labeled data but without conf')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--noise', type=float, default=0.0, metavar='N')
parser.add_argument('--eval-epochs', type=int, default=3, metavar='E',
                    help='epochs to evaluate model')
parser.add_argument('--prior', type=float, default=0.2,
                    help='ratio of confidence data')
parser.add_argument('--traj-size', type=int, default=2000)
parser.add_argument('--ofolder', type=str, default='log')
parser.add_argument('--ifolder', type=str, default='demonstrations')
args = parser.parse_args()

env = gym.make(args.env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

plabel = ''
try:
    expert_traj = np.load("./{}/{}_mixture.npy".format(args.ifolder, args.env))
    expert_conf = np.load("./{}/{}_mixture_conf.npy".format(args.ifolder, args.env))
    expert_conf += (np.random.randn(*expert_conf.shape) * args.noise)
    expert_conf = np.clip(expert_conf, 0.0, 1.0)
except:
    print('Mixture demonstrations not loaded successfully.')
    assert False

idx = np.random.choice(expert_traj.shape[0], args.traj_size, replace=False)
expert_traj = expert_traj[idx, :]
expert_conf = expert_conf[idx, :]


##### semi-confidence learning #####
num_label = int(args.prior * expert_conf.shape[0])

p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx, :]
expert_conf = expert_conf[p_idx, :]

if not args.only and args.weight:

    labeled_traj = torch.Tensor(expert_traj[:num_label, :]).to(device)
    unlabeled_traj = torch.Tensor(expert_traj[num_label:, :]).to(device)
    label = torch.Tensor(expert_conf[:num_label, :]).to(device)

    classifier = Classifier(expert_traj.shape[1], 40).to(device)
    optim = optim.Adam(classifier.parameters(), 3e-4, amsgrad=True)
    cu_loss = CULoss(expert_conf, beta=1-args.prior, non=True) 

    batch = min(128, labeled_traj.shape[0])
    ubatch = int(batch / labeled_traj.shape[0] * unlabeled_traj.shape[0])
    iters = 25000
    for i in range(iters):
        l_idx = np.random.choice(labeled_traj.shape[0], batch)
        u_idx = np.random.choice(unlabeled_traj.shape[0], ubatch)

        labeled = classifier(Variable(labeled_traj[l_idx, :]))
        unlabeled = classifier(Variable(unlabeled_traj[u_idx, :]))
        smp_conf = Variable(label[l_idx, :])

        optim.zero_grad()
        risk = cu_loss(smp_conf, labeled, unlabeled)
        
        risk.backward()
        optim.step()
       
        if i % 1000 == 0:
            print('iteration: {}\tcu loss: {:.3f}'.format(i, risk.data.item()))

    classifier = classifier.eval()
    expert_conf = torch.sigmoid(classifier(torch.Tensor(expert_traj).to(device))).detach().cpu().numpy()
    expert_conf[:num_label, :] = label.cpu().detach().numpy()
elif args.only and args.weight:
    expert_traj = expert_traj[:num_label, :]
    expert_conf = expert_conf[:num_label, :]
    if args.noconf:
        expert_conf = np.ones(expert_conf.shape)