import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from glob import glob
from config import cfg

data_dir = cfg['data']['data_dir']
preds = glob('save_img/*.*')
preds.sort()
labels = glob('%s/test/*.*' % data_dir)
labels.sort()

psnrs = 0
ssims = 0
for low, high in zip(preds, labels):
    low = cv2.imread(low)
    high = cv2.imread(high)
    psnrs += peak_signal_noise_ratio(low, high)
    ssims += structural_similarity(low, high, data_range=255, multichannel=True)

print('average psnr: ', psnrs / len(preds))
print('average ssim: ', ssims / len(preds))
