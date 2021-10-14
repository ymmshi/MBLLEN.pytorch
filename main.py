import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.model import MBLLEN
from utils.dataset import MBLLENData
from utils.loss import Loss


class Model(LightningModule):
    def __init__(self, model_cfg):
        super(Model, self).__init__()
        em_channel = model_cfg['em_channel']
        fem_channel = model_cfg['fem_channel'] 
        block_num = model_cfg['block_num']
        self.model = MBLLEN(em_channel, fem_channel, block_num)
        self.compute_loss = Loss(self.log)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, label = batch
        pred = self(input)
        loss = self.compute_loss(pred, label, 'train')
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def validation_step(self, batch, batch_idx):
        input, label = batch
        pred = self(input)
        loss = self.compute_loss(pred, label, 'val')
        self.log("val_loss", loss)

    def on_epoch_end(self):
        pass


class Data(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_dir = data_cfg['data_dir']
        self.batch_size = data_cfg['batch_size']
        self.num_workers = data_cfg['num_workers']
        self.dark_or_low = data_cfg['dark_or_low']
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MBLLENData(self.data_dir, 
                                mode1='train', 
                                mode2=self.dark_or_low, 
                                transform=self.transform)
        self.val_dataset = MBLLENData(self.data_dir, 
                                mode1='test', 
                                mode2=self.dark_or_low, 
                                transform=self.transform)      
        self.test_dataset = MBLLENData(self.data_dir, 
                                mode1='test', 
                                mode2=self.dark_or_low, 
                                transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

        
if __name__ == '__main__':
    from config import cfg
    model = Model(cfg['model'])
    data = Data(cfg['data'])
    trainer = Trainer(gpus=cfg['trainer']['gpus'], 
                      max_epochs=cfg['trainer']['max_epochs'], 
                      accelerator='ddp', 
                      precision=cfg['trainer']['precision'],
                      progress_bar_refresh_rate=1, 
                      plugins=DDPPlugin(find_unused_parameters=False), 
                      callbacks=[ModelCheckpoint(monitor=cfg['trainer']['monitor']), LearningRateMonitor(logging_interval='step')])
    trainer.fit(model, data)