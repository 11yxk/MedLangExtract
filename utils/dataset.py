import json
import os
import torch
import pandas as pd
from monai.transforms import ( Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):

    def __init__(self, json_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224,224], 
                 progressive_level='P5'):
        """
        初始化数据集
        Args:
            json_path: JSON文件路径
            root_path: 图像根目录路径
            tokenizer: 分词器路径
            mode: 模式 ('train', 'valid', 'test')
            image_size: 图像尺寸
            progressive_level: progressive text级别 ('P0'-'P5')
        """
        super(QaTa, self).__init__()

        self.mode = mode
        self.progressive_level = progressive_level
        
        # 读取JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 提取图像和文本信息
        self.image_list = [item['img_name'] for item in self.data]
        self.mask_list = [item['mask_name'] for item in self.data]
        self.text_data = self.data  # 保存完整的文本数据

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)

        # 获取图像和mask路径
        image = os.path.join(self.root_path, 'images', self.image_list[idx])
        gt = os.path.join(self.root_path, 'masks', self.mask_list[idx])
        
        # 根据progressive_level获取文本
        text_item = self.text_data[idx]
        caption = text_item[self.progressive_level]['text']

        # 处理文本tokenization
        token_output = self.tokenizer.encode_plus(
            caption, 
            padding='max_length',
            max_length=64,  # 增加最大长度以适应更长的描述
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt == 255, 1, 0)
        
        # Ensure GT is single channel (take first channel if multi-channel)
        if gt.shape[0] == 3:
            gt = gt[0:1]  # Take only the first channel
            
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)} 

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])

        return trans
