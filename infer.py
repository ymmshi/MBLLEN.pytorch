import torch
from glob import glob
from PIL import Image
from torchvision import transforms
from main import Model
from config import cfg
from torchvision.utils import save_image
import os
from tqdm import tqdm

def infer(model, img):
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0).cuda()
    output = model(img)
    return img, output


if __name__ == '__main__':
    data_dir = cfg['data']['data_dir']
    paths = glob(data_dir + '/test_lowlight/*.*')
    transform = transforms.Compose([transforms.ToTensor()])
    
    model = Model(cfg['model'])
    model.load_state_dict(torch.load('pretrained_models/lowlight.ckpt')['state_dict'])
    model = model.cuda()
    save_dir = './save_img'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for path in tqdm(paths):
            img = Image.open(path)
            im_in, im_out = infer(model, img)
            save_image(torch.cat([im_out], dim=3), os.path.join(save_dir, os.path.basename(path) + '.png'), normalize=True, value_range=(0, 1))

