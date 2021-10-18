cfg = {
    'model': {},
    'data': {},
    'optimizer': {},
    'trainer': {},
}
cfg['model'] = {
    'em_channel': 8,
    'fem_channel': 32,
    'block_num': 9,
}
cfg['data'] = {
    'data_dir': '/home/ymshi/Face/data/MBLLEN_dataset',
    'batch_size': 16,
    'num_workers': 4,
    'dark_or_low': 'lowlight'
}
cfg['trainer'] = {
    'gpus':[0,1],
    'precision': 32,
    'max_epochs': 80,
    'monitor': 'val_loss'
}