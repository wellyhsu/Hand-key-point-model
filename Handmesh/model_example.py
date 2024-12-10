import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from mobrecon.models.densestack import DenseStack_Backnone
from mobrecon.models.modules import Reg2DDecode3D
from mobrecon.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from mobrecon.build import MODEL_REGISTRY
from mobrecon.main import setup
from options.cfg_options import CFGOptions
from thop import clever_format, profile
from torchinfo import summary
import pandas as pd

#from models.imagenet import mobilenetv2

#net = mobilenetv2()
#net.load_state_dict(torch.load('pretrained/mobilenetv2-c5e733a8.pth'))

@MODEL_REGISTRY.register()
# Replace model here
class MobRecon_DS(nn.Module):
    def __init__(self, cfg, control=None):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg
        self.control = control  # 將 control 儲存為實例變數

        # 2D encoding - backbone
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
                                            kpts_num=cfg.MODEL.KPTS_NUM, control=self.control)
        # 獲取當前執行 Python 腳本所在的目錄路徑
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, 'template/template.ply')
        transform_fp = os.path.join(cur_dir, 'template', 'transform.pkl')

        # 生成、載入或處理與「螺旋結構」相關的轉換矩陣（transform matrices）和索引資料，主要用於 3D 網格（如手部模型）的處理
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp,
                                                                template_fp,
                                                                cfg.MODEL.SPIRAL.DOWN_SCALE,
                                                                cfg.MODEL.SPIRAL.LEN,
                                                                cfg.MODEL.SPIRAL.DILATION)
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        
        # 2D lifting to 3D + 3D decoder
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                       meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'])

    def forward(self, x):
        if x.size(1) == 6:
            pred3d_list = []
            pred2d_pt_list = []
            for i in range(2):
                # 獲得2D座標
                latent, pred2d_pt = self.backbone(x[:, 3*i:3*i+3])
                # 3D decoder
                pred3d = self.decoder3d(pred2d_pt, latent)

                pred3d_list.append(pred3d)
                pred2d_pt_list.append(pred2d_pt)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)
            pred3d = torch.cat(pred3d_list, -1)
        else:
            latent, pred2d_pt = self.backbone(x)
            pred3d = self.decoder3d(pred2d_pt, latent)

        return {'verts': pred3d,
                'joint_img': pred2d_pt
                }

args = CFGOptions().parse()
args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
cfg = setup(args)

# 定義模型並傳遞 cfg
# 選擇要計算的model版本: Densestack, mobilenet_v3
model = MobRecon_DS(cfg, control='mobilenet_v3')

# 修改 input 的大小，符合模型的輸入
input = torch.randn(1, 6, 128, 128)


