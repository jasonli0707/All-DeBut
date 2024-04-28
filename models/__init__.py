from .resnet import resnet110
from .resnet_debut import resnet110_debut 
from .resnet_butterfly import resnet110_butterfly
from .resnet_svd import resnet110_svd
from .resnet_auto import resnet110_auto
from .vgg import vgg16_bn
from .vgg_debut import vgg16_bn_debut
from .vgg_butterfly import vgg16_bn_butterfly
from .vgg_svd import vgg16_bn_svd
# from .vgg_fastfood import vgg16_bn_fastfood
from .vgg_auto import vgg16_bn_auto
from .pointnet import pointnet, pointnet_loss
from .pointnet_debut import pointnet_debut, pointnet_debut_loss
from .pointnet_svd import pointnet_svd, pointnet_svd_loss
from .pointnet_butterfly import pointnet_butterfly, pointnet_butterfly_loss
# from .pointnet_fastfood import pointnet_fastfood, pointnet_fastfood_loss

model_dict = {
    'resnet110': resnet110,
    'resnet110_debut': resnet110_debut,
    'resnet110_butterfly': resnet110_butterfly,
    'resnet110_svd': resnet110_svd,
    'resnet110_auto': resnet110_auto,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg16_bn_debut': vgg16_bn_debut,
    'vgg16_bn_butterfly': vgg16_bn_butterfly,
    'vgg16_bn_svd': vgg16_bn_svd,
    # 'vgg16_bn_fastfood': vgg16_bn_fastfood,
    'vgg16_bn_auto': vgg16_bn_auto,
    'pointnet': [pointnet, pointnet_loss],
    'pointnet_debut': [pointnet_debut, pointnet_debut_loss],
    'pointnet_svd': [pointnet_svd, pointnet_svd_loss],
    'pointnet_butterfly': [pointnet_butterfly, pointnet_butterfly_loss],
    # 'pointnet_fastfood': [pointnet_fastfood, pointnet_fastfood_loss]
}
