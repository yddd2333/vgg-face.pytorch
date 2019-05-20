import torch
import torch.nn as nn
from vgg_face_dag import vgg_face_dag

class DIMNet(nn.Module):
    def __init__(self, pretrained, num_class):
        super(DIMNet, self).__init__()
        self.base_v = vgg_face_dag(pretrained)
        self.base_c = vgg_face_dag(pretrained)

        # settings
        cls_dims = 512
        vgg_out_dim = 7*7*512
        decompose_dim_in = 512
        self.high_dim_v = nn.Sequential(nn.Linear(vgg_out_dim, decompose_dim_in), nn.ReLU())
        self.high_dim_c = nn.Sequential(nn.Linear(vgg_out_dim, decompose_dim_in), nn.ReLU())

        self.identity = nn.Linear(cls_dims, num_class)
        self.gender = nn.Linear(cls_dims, 2)


    def forward(self, c, v):
        v = self.base_v(v)
        v = v.view(v.size(0), -1)
        v = self.high_dim_v(v)

        c = self.base_c(c)
        c = c.view(c.size(0), -1)
        c = self.high_dim_c(c)

        # concat
        embeddin_vc = torch.cat((v, c), 0)
        identity_out = self.identity(embeddin_vc)
        gender_out = self.gender(embeddin_vc)

        logit_list = [identity_out, gender_out]

        return logit_list

if __name__ == "__main__":
    x = torch.randn(5, 3, 224, 224)
    DIMNet_model = DIMNet('./vgg_face_dag.pth', 10)
    output = DIMNet_model(x, x)
    print (output[1].size())
    print (output)




