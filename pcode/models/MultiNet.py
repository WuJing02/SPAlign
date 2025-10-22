import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TransferNet']

class TransferNet(nn.Module):
    def __init__(self, models, query_weight=None, key_weight=None, weight_vector=None, input_channel=64, factor=8):
        super().__init__()
        self.num_models = len(models)
        
        for i, model in enumerate(models):
            setattr(self, 'model'+str(i), model)
        
        self.attn = torch.zeros(self.num_models).cuda()
        self.save_attn = False
        self.attn_matrix = torch.zeros(self.num_models).unsqueeze(0).cuda()

    def forward(self, x):                       
        pro = self.model0(x)

        key = pro[:, None, :]
        pro = pro.unsqueeze(-1)

        for i in range(1, self.num_models-1):
            temp_pro = getattr(self, 'model'+str(i))(x)
            key = torch.cat([key, temp_pro[:,None,:]], 1)
            temp_pro = temp_pro.unsqueeze(-1)
            pro = torch.cat([pro, temp_pro],-1)
        
        temp_pro = getattr(self, 'model'+str(self.num_models-1))(x)
        
        with torch.no_grad():
            query_stu = temp_pro[:, None, :]
            key = torch.cat([key, temp_pro[:,None,:]], 1)
            pro = torch.cat([pro, temp_pro.unsqueeze(-1)],-1)
            energy_stu = torch.bmm(query_stu, key.permute(0,2,1))
            energy_stu_pos = F.relu(energy_stu)
            attn_stu = energy_stu_pos / torch.sum(energy_stu_pos, dim=-1, keepdim=True)
        
            if self.save_attn:
                avg = torch.mean(energy_stu, dim = 0).squeeze()
                self.attn += avg
                self.attn_matrix = torch.cat([self.attn_matrix, avg[None,:]],0)

            attn_target_stu = torch.bmm(pro, attn_stu.permute(0,2,1)).squeeze(-1)
            avg_logit = 0
            for i in range(self.num_models):
                avg_logit += pro[:,:,i]

        return temp_pro, attn_target_stu
