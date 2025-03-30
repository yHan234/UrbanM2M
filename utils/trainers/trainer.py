from typing import List
import os
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from ..utils import *
from tensorboardX import SummaryWriter as sw


class Trainer:
    def __init__(self, model, args, dataset, device):
        self.val_loader = None
        self.train_loader = None
        self.model = model
        self.args = args
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                          eps=1e-3)
        self.amp_scaler = GradScaler(enabled=args.use_mix)
        self.scheduler = get_scheduler(self.optimizer, args, args.epochs)
        self.criterion = nn.MSELoss(size_average=False, reduce=True).to(self.device)
        
        self.dataset = dataset
        self.get_loader()

        self.eta = 1
        self.enc_len = args.enc_len
        self.fore_len = args.fore_len
        self.batch_size = args.batch_size

        self.summary_train = sw('/root/tf-logs/train', self.args.model_name)
        self.summary_val = sw('/root/tf-logs/val', self.args.model_name)

        self.model_dir = args.model_dir
        self.model_name = args.model_name

        self.start_epoch = 0
        self.total_epoch = args.epochs
        self.resume = None
        if self.resume:
            checkpoint = torch.load(self.resume)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.eta = checkpoint['eta']
            self.model_name = self.resume.split('/')[-1].split('-epoch')[0].strip()

    def get_teacher_masker(self, x: torch.Tensor) -> List[torch.Tensor]:
        random_flip = np.random.random_sample((self.enc_len - 1, x.shape[0]))
        true_token = (random_flip < self.eta)
        one = torch.FloatTensor(1, x[0].size(1), x[0].size(
            2), x[0].size(3)).fill_(1.0).to(self.device)
        zero = torch.FloatTensor(1, x[0].size(1), x[0].size(
            2), x[0].size(3)).fill_(0.0).to(self.device)
        masks = []
        for t in range(self.fore_len - 1):
            masks_b = []
            for i in range(x.shape[0]):
                if true_token[t, i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batch size
            masks.append(mask)
        return masks

    def get_loader(self):

        n_val = int(len(self.dataset) * self.args.val_prop)
        n_train = len(self.dataset) - n_val

        train_set, val_set = random_split(self.dataset, [n_train, n_val])

        self.train_loader = DataLoader(
            train_set, shuffle=True, batch_size=self.args.batch_size, num_workers=8)
        self.val_loader = DataLoader(
            val_set, shuffle=False, batch_size=self.args.batch_size, num_workers=8)

    def scaler_step(self, loss) -> None:
        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        self.optimizer.zero_grad()

    def loop(self):
        for epoch in range(self.start_epoch, self.total_epoch):
            loop_loss = self.train_loop(self.train_loader, epoch)
            self.eta -= self.args.eta_decay
            self.eta = max(self.eta, 0)
            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'eta': self.eta,
                'loss': loop_loss
            }, epoch)
            if epoch % 5 == 0:
                self.val_loop(self.val_loader)
            self.scheduler.step()
        self.summary_train.close()

    def train_loop(self, loader: DataLoader, epoch_count: int):
        tot_loss = 0
        self.model.train()
        batch_count = len(loader.dataset)//self.args.batch_size
        with tqdm(total=batch_count, ncols=75) as pbar:
            for rc, spa_vars, x in loader:
                if self.args.use_mix:
                    with autocast():
                        loss = self.model_forward(x, spa_vars)
                else:
                    loss = self.model_forward(x, spa_vars)
                self.scaler_step(loss)
                pbar.update(1)
                pbar.set_postfix(
                    {'loss': loss.item(), 'avg loss': tot_loss / pbar.n})
                tot_loss += float(loss.item())
        print(f'finish train epoch {epoch_count}')
        return float(tot_loss/batch_count)

    def val_loop(self, loader: DataLoader):
        tot_loss = 0
        self.model.eval()
        with tqdm(total=len(loader.dataset)//self.args.batch_size, ncols=75) as pbar:
            with torch.no_grad():
                for rc, spa_vars, x in loader:
                    loss = self.model_forward(x, spa_vars, [])
                    pbar.update(1)
                    pbar.set_postfix(
                        {'loss': loss.item(), 'avg loss': tot_loss / pbar.n})
                    tot_loss += float(loss.item())
        print('finish validation')

    def model_forward(self, x0: torch.Tensor, spa_vars: torch.Tensor, mask: list = []):
        # torch.cuda.empty_cache()
        spa_vars = spa_vars.to(self.device)
        x = x0.to(self.device)
        if not mask:
            mask = self.get_teacher_masker(x)
        # input  00 01 02 03 04 05
        # output 06 07 08 09 10 11
        # gn_imgs = self.model(x, spa_vars, mask)[:, -self.fore_len:]
        # gt_imgs = x[:, -self.fore_len:]
        gn_imgs = self.model(x, spa_vars, mask)

        gt_imgs = x[:, 1:]
        loss = self.criterion(gn_imgs, gt_imgs)
        return loss

    def save_checkpoint(self, checkpoint: dict, epoch_label: int) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # if epoch_label%8==0:
        #     checkpoint_filename = f'{self.model_name}-e{epoch_label}.pth'
        #     checkpoint_path = os.path.join(self.model_dir, checkpoint_filename)
        #     torch.save(checkpoint, checkpoint_path)

        model_filename = f'{self.model_name}-e{epoch_label}.pth'
        model_path = os.path.join(self.model_dir, model_filename)
        torch.save(self.model, model_path)
        # script_model = torch.jit.script(self.model)
        # script_model.save(model_path)
