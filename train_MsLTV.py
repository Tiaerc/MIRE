import logging
import os.path
import argparse

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config.features import ALL_FEATURES, ACTIVE_FEATURES
from model.MsLTV import MSModel
from utils.common import get_root_path
from utils.datasets import BaseDataset, base_collate
from utils.run import EarlyStopping, Trainer, set_seed

logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                  handlers=[logging.StreamHandler()]
                  )

parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--rec_loss_weight', type=float, default=0.1)
parser.add_argument('--orthogonal_loss_weight', type=float, default=0.)
parser.add_argument('--cl_loss_weight_us', type=float, default=10)
parser.add_argument('--cl_loss_weight_s', type=float, default=0.)
parser.add_argument('--kl_weight', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda:0')
# 解析参数
args = parser.parse_args()



def train(jf_game_id, train_start_date, train_end_date, val_start_date, val_end_date, save_path, features, model='DNN', device='cuda:0', max_row=None, minus_label_col=None):
  # args
  epochs = 8
  batch_size = 1024
  lr = args.lr
  lr_scheduler_patience = 0
  early_stop_patience = 2

  # log feature info
  feature_number = {
      'dense': len(features['dense']),
      'sparse': len(features['sparse']),
      'seq': len(features['seq'])
  }
  logging.info(f'feature number:{feature_number}')

  # model
  model = MultiViewModel(features=features,
                         hidden_unit=[128],
                         encode_dim=64,
                         rec_loss_weight=args.rec_loss_weight,
                         orthogonal_loss_weight=args.orthogonal_loss_weight,
                         cl_loss_weight_us=args.cl_loss_weight_us,
                         cl_loss_weight_s=args.cl_loss_weight_s,
                         kl_weight = args.kl_weight,
                         device=args.device)

  # optim
  optimizer = optim.Adam(model.parameters(), lr)
  scheduler = None
  early_stopping = EarlyStopping(patience=early_stop_patience)

  # data
  train_dataset = BaseDataset(jf_game_id, train_start_date, train_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=base_collate)
  val_dataset = BaseDataset(jf_game_id, val_start_date, val_end_date, features=features, max_rows=max_row, minus_label_col=minus_label_col)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=base_collate)

  # train
  trainer = Trainer(model, optimizer, scheduler, early_stopping)
  trainer.train(save_path, epochs, train_loader, val_loader)


if __name__ == '__main__':
  set_seed(42)
  for jf_game_id in ['XXX']:
      save_path = os.path.join(get_root_path(), 'results', f'{jf_game_id}_test_rec{args.rec_loss_weight}_clus{args.cl_loss_weight_us}_kl{args.kl_weight}')
      train(jf_game_id, 805, 818, 819, 820,
            save_path=save_path, features=ALL_FEATURES, minus_label_col='ltv_6h', max_row=None)

