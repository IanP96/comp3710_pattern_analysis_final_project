"""
Code for model components (classes/functions)
"""

"""
MODULE.PY DOCSTRING STARTS HERE

Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen (robinbg@foxmail.com)

-----------------------------

model.py: Network Modules

(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator

TIMEGAN.PY DOCSTRING STARTS HERE

Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import logging
import random
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray

from dataset import batch_generator
from utils import extract_time, random_generator, norm_min_max, TORCH_DEVICE, kl_metric
from constants import (
    WEIGHTS_DIR,
    OUTPUT_DIR,
    RANGPUR_NUM_TRAINING_ITERATIONS,
    LOCAL_NUM_TRAINING_ITERATIONS,
    RANGPUR_VALIDATE_INTERVAL,
    LOCAL_VALIDATE_INTERVAL,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

    Args:
      - input: input time-series features. (L, N, X) = (24, ?, 6)
      - h3: (num_layers, N, H). [3, ?, 24]

    Returns:
      - H: embeddings
    """

    def __init__(self, opt):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer
        )
        # self.norm = nn.BatchNorm1d(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """

    def __init__(self, opt):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim, hidden_size=opt.z_dim, num_layers=opt.num_layer
        )

        #  self.norm = nn.BatchNorm1d(opt.z_dim)
        self.fc = nn.Linear(opt.z_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer
        )
        # self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        # g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """

    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
        )
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        #  s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer,
        )
        # self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat


class TimeGAN:
    """TimeGAN Class"""

    @property
    def name(self):
        return "TimeGAN"

    def seed(self, seed_value: int) -> None:
        """
        Seed all functionality.

        Args:
            seed_value (int): seed to use, or -1 to avoid using a manual seed
        """

        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def __init__(
        self,
        opt: Namespace,
        ori_data: NDArray[np.float32],
        validate_data: NDArray[np.float32],
        test_data: NDArray[np.float32],
        load_weights=False,
    ):

        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalise variables
        self.opt = opt
        self.ori_data, self.min_val, self.max_val = norm_min_max(ori_data)
        self.validate_data = validate_data
        self.test_data = test_data
        assert len(self.validate_data.shape) == len(self.test_data.shape) == 2
        self.validate_min_val, self.validate_max_val = (
            np.min(self.validate_data, 0),
            np.max(self.validate_data, 0) - np.min(self.validate_data, 0),
        )
        self.test_min_val, self.test_max_val = (
            np.min(self.test_data, 0),
            np.max(self.test_data, 0) - np.min(self.test_data, 0),
        )
        logger.debug(
            "From normalising, got min_val: %s, max_val: %s", self.min_val, self.max_val
        )
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)
        self.data_num, _, _ = np.asarray(ori_data).shape  # 3661; 24; 6
        # self.trn_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        # self.tst_dir = os.path.join(self.opt.outf, self.opt.name, "test")
        self.device = TORCH_DEVICE

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        # Create and initialize networks.
        self.nete = Encoder(self.opt).to(self.device)
        self.netr = Recovery(self.opt).to(self.device)
        self.netg = Generator(self.opt).to(self.device)
        self.netd = Discriminator(self.opt).to(self.device)
        self.nets = Supervisor(self.opt).to(self.device)

        self.num_iterations: int
        self.validate_interval: int
        if self.opt.env == "rangpur":
            self.num_iterations = RANGPUR_NUM_TRAINING_ITERATIONS
            self.validate_interval = RANGPUR_VALIDATE_INTERVAL
        elif self.opt.env == "local":
            self.num_iterations = LOCAL_NUM_TRAINING_ITERATIONS
            self.validate_interval = LOCAL_VALIDATE_INTERVAL
        else:
            raise ValueError(f"Unknown environment: {self.opt.env}")

        weights_path = Path(OUTPUT_DIR, WEIGHTS_DIR)
        if load_weights and weights_path.exists():
            # assert weights_path.exists(), f"Weights path {weights_path} does not exist."
            logger.info(
                "Loading pre-trained networks from directory: %s ...", weights_path
            )
            # self.opt.iteration = torch.load(Path(weights_path, "netG.pth"))["epoch"]
            self.nete.load_state_dict(
                torch.load(Path(weights_path, "netE.pth"))["state_dict"]
            )
            self.netr.load_state_dict(
                torch.load(Path(weights_path, "netR.pth"))["state_dict"]
            )
            self.netg.load_state_dict(
                torch.load(Path(weights_path, "netG.pth"))["state_dict"]
            )
            self.netd.load_state_dict(
                torch.load(Path(weights_path, "netD.pth"))["state_dict"]
            )
            self.nets.load_state_dict(
                torch.load(Path(weights_path, "netS.pth"))["state_dict"]
            )
            logger.info("Finished loading pre-trained networks.")
        else:
            logger.info("Not using pre-trained weights. Training from scratch.")

        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        # Setup optimizer
        if self.opt.isTrain:
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()
            self.optimizer_e = optim.Adam(
                self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_r = optim.Adam(
                self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_g = optim.Adam(
                self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_d = optim.Adam(
                self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            self.optimizer_s = optim.Adam(
                self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )

    def forward_e(self):
        """Forward propagate through netE"""
        self.H = self.nete(self.X)

    def forward_er(self):
        """Forward propagate through netR"""
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        """Forward propagate through netG"""
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)

    def forward_dg(self):
        """Forward propagate through netD"""
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
        """Forward propagate through netG"""
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        """Forward propagate through netS"""
        self.H_supervise = self.nets(self.H)
        # print(self.H, self.H_supervise)

    def forward_sg(self):
        """Forward propagate through netS"""
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        """Forward propagate through netD"""
        self.Y_real = self.netd(self.H)
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def backward_er(self):
        """Backpropagate through netE"""
        self.err_er = self.l_mse(self.X_tilde, self.X)
        self.err_er.backward(retain_graph=True)
        # print("Loss: ", self.err_er)
        logger.debug("Loss ER: %s", self.err_er.item())

    def backward_er_(self):
        """Backpropagate through netE"""
        self.err_er_ = self.l_mse(self.X_tilde, self.X)
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
        self.err_er.backward(retain_graph=True)
        # print("Loss: ", self.err_er_, self.err_s)

    def backward_g(self):
        """Backpropagate through netG"""
        self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))

        self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
        self.err_g_V1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6)
                - torch.sqrt(torch.std(self.X, [0])[1] + 1e-6)
            )
        )  # |a^2 - b^2|
        self.err_g_V2 = torch.mean(
            torch.abs((torch.mean(self.X_hat, [0])[0]) - (torch.mean(self.X, [0])[0]))
        )  # |a - b|
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_g = (
            self.err_g_U
            + self.err_g_U_e * self.opt.w_gamma
            + self.err_g_V1 * self.opt.w_g
            + self.err_g_V2 * self.opt.w_g
            + torch.sqrt(self.err_s)
        )
        self.err_g.backward(retain_graph=True)
        logger.debug("Loss G: %s", self.err_g.item())

    def backward_s(self):
        """Backpropagate through netS"""
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)
        logger.debug("Loss S: %s", self.err_s.item())
        # print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
        """Backpropagate through netD"""
        self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
        self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
        self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
        self.err_d = (
            self.err_d_real + self.err_d_fake + self.err_d_fake_e * self.opt.w_gamma
        )
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)
        # print("Loss D: ", self.err_d)
        logger.debug("Loss D: %s", self.err_d.item())

    def optimize_params_er(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_er()
        self.forward_s()
        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s()
        self.optimizer_s.step()

    def optimize_params_g(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_s()
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
        self.optimizer_s.step()

    def optimize_params_d(self):
        """Forwardpass, Loss Computation and Backwardpass."""
        # Forward-pass
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

    def save_weights(self):
        """Save network weights."""

        weight_dir = Path(OUTPUT_DIR, WEIGHTS_DIR)
        logger.info("Saving network weights to directory: %s ...", weight_dir)
        if not weight_dir.exists():
            weight_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory to save weights.")

        # torch.save(
        #     {"epoch": epoch + 1, "state_dict": self.nete.state_dict()},
        #     Path(weight_dir, "netE.pth"),
        # )
        # torch.save(
        #     {"epoch": epoch + 1, "state_dict": self.netr.state_dict()},
        #     Path(weight_dir, "netR.pth"),
        # )
        # torch.save(
        #     {"epoch": epoch + 1, "state_dict": self.netg.state_dict()},
        #     Path(weight_dir, "netG.pth"),
        # )
        # torch.save(
        #     {"epoch": epoch + 1, "state_dict": self.netd.state_dict()},
        #     Path(weight_dir, "netD.pth"),
        # )
        # torch.save(
        #     {"epoch": epoch + 1, "state_dict": self.nets.state_dict()},
        #     Path(weight_dir, "netS.pth"),
        # )
        torch.save(
            {"state_dict": self.nete.state_dict()},
            Path(weight_dir, "netE.pth"),
        )
        torch.save(
            {"state_dict": self.netr.state_dict()},
            Path(weight_dir, "netR.pth"),
        )
        torch.save(
            {"state_dict": self.netg.state_dict()},
            Path(weight_dir, "netG.pth"),
        )
        torch.save(
            {"state_dict": self.netd.state_dict()},
            Path(weight_dir, "netD.pth"),
        )
        torch.save(
            {"state_dict": self.nets.state_dict()},
            Path(weight_dir, "netS.pth"),
        )
        logger.info("Weights saved.")

    def train_one_iter_er(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er()

    def train_one_iter_er_(self):
        """Train the model for one epoch."""

        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er_()

    def train_one_iter_s(self):
        """Train the model for one epoch."""

        # self.nete.eval()
        self.nets.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train supervisor
        self.optimize_params_s()

    def train_one_iter_g(self):
        """Train the model for one epoch."""

        # self.netr.eval()
        # self.nets.eval()
        # self.netd.eval()

        self.netg.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.opt.seq_len)

        # train supervisor
        self.optimize_params_g()

    def train_one_iter_d(self):
        """Train the model for one epoch."""

        # self.nete.eval()
        # self.netr.eval()
        # self.nets.eval()
        # self.netg.eval()

        self.netd.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.opt.seq_len)

        # train supervisor
        self.optimize_params_d()

    def train_and_generate(self):
        """Train the model and generate some synthetic data"""

        logger.info("Starting training ...")

        for iter in range(self.num_iterations):
            # Train for one iter
            self.train_one_iter_er()
            logger.info("Encoder training step: %s/%s", iter + 1, self.num_iterations)

        for iter in range(self.num_iterations):
            # Train for one iter
            self.train_one_iter_s()
            logger.debug(
                "Supervisor training step: %s/%s", iter + 1, self.num_iterations
            )

        last_kl_spread = float("inf")
        last_kl_mpr = float("inf")
        increase_count = 0
        for iter in range(self.num_iterations):
            # Train for one iter
            for kk in range(2):
                self.train_one_iter_g()
                self.train_one_iter_er_()
            self.train_one_iter_d()
            logger.debug(
                "Supervisor training step: %s/%s", iter + 1, self.num_iterations
            )
            if iter % self.validate_interval == 0:
                generated_data = self.generation(
                    len(self.validate_data),
                    self.validate_max_val,
                    self.validate_min_val,
                )
                logger.debug("Generated data shape: %s", generated_data.shape)
                logger.debug("Validation data shape: %s", self.validate_data.shape)
                try:
                    kl_spread = kl_metric(self.validate_data, generated_data, "spread")
                    kl_mpr = kl_metric(self.validate_data, generated_data, "mpr")
                except AssertionError:
                    logger.exception("KL metric computation failed, got error:")
                    continue
                logger.info(
                    "KL Spread: %s, KL mid-price return: %s", kl_spread, kl_mpr
                )
                # todo add ask > bid as metric
                if kl_spread > last_kl_spread or kl_mpr > last_kl_mpr:
                    increase_count += 1
                else:
                    increase_count = 0
                last_kl_spread = kl_spread
                last_kl_mpr = kl_mpr

                # if increase_count >= 3:
                #     logger.info("Early stopping at iteration %s", iter)
                #     break

                self.save_weights()

        logger.info("Training finished.")

        self.save_weights()

    def run_inference(self):
        self.generated_data = self.generation(
            len(self.test_data), self.test_max_val, self.test_min_val
        )
        GENERATED_DATA_PATH = Path(OUTPUT_DIR, "generated_data.npy")
        np.save(GENERATED_DATA_PATH, self.generated_data)
        logger.info(
            "Finished synthetic data generation and written synthetic data to %s",
            GENERATED_DATA_PATH,
        )

    def generation(
        self, num_rows: int, max_val: NDArray, min_val: NDArray, mean=0.0, std=1.0
    ) -> NDArray[np.float32]:

        assert num_rows > 0, "num_samples should be a positive integer."

        # Synthetic data generation
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size
        )
        num_batches = num_rows // self.max_seq_len
        # todo figure this out
        self.Z = random_generator(
            num_batches, self.opt.z_dim, self.opt.seq_len, mean, std
        )
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.E_hat = self.netg(self.Z)  # [?, 24, 24]
            self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
            generated_data_curr: NDArray = (
                self.netr(self.H_hat).cpu().detach().numpy()
            )  # [?, 24, 24]

        generated_data = np.empty(
            (num_batches, self.max_seq_len, self.opt.z_dim), dtype=np.float32
        )
        for i in range(num_batches):
            temp = generated_data_curr[i, : self.ori_time[i], :]
            generated_data[i] = temp

        # Renormalisation
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        # Reshape to 2D
        generated_data_2d = generated_data.reshape(-1, generated_data.shape[2])
        return generated_data_2d
