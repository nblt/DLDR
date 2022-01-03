# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from models.networks import get_norm_layer, NLayerDiscriminator, NLayerSNDiscriminator, PixelDiscriminator, init_weights
from models.image_pool import ImagePool
from einops import rearrange


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num=0):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_SN':
        net = NLayerSNDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    init_weights(net, init_type, init_gain=init_gain)
    return net

class PretrainG(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        B, N, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x_full = self.decoder(x_full) # [B, N_mask, 3 * 16 * 16]
        rec_patches = x_full[:, -pos_emd_mask.shape[1]:]

        return x_full, rec_patches

class FailGANPretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 gan_mode='lsgan',
                 input_nc=3, 
                 ndf=64, 
                 netD='basic', 
                 n_layers_D=3,
                 pool_size=10000,
                 lambda_gan=2.0,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.netG = PretrainG(
            img_size=img_size, 
            patch_size=patch_size, 
            encoder_in_chans=encoder_in_chans, 
            encoder_num_classes=encoder_num_classes, 
            encoder_embed_dim=encoder_embed_dim, 
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads, 
            decoder_num_classes=decoder_num_classes, 
            decoder_embed_dim=decoder_embed_dim, 
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.netD = define_D(
            input_nc=input_nc, 
            ndf=ndf,
            netD=netD,
            n_layers_D=n_layers_D,
        )

        # gan set
        self.patch_size = patch_size
        target_real_label=1.0
        target_fake_label=-1.0
        target_gene_label=0.0
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('gene_label', torch.tensor(target_gene_label))
        self.criterionRec = nn.MSELoss()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.GANLoss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.GANLoss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.GANLoss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        self.fake_pool = ImagePool(pool_size)
        self.lambda_gan = lambda_gan

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real == 'real':
            target_tensor = self.real_label
        elif target_is_real == 'fake':
            target_tensor = self.fake_label
        elif target_is_real == 'gene':
            target_tensor = self.gene_label
        else:
            raise RuntimeError(f"target tensor wrong {target_is_real}")
        return target_tensor.expand_as(prediction)

    def criterionGAN(self, prediction, target_is_real):
        # print(f"pred {prediction.shape}")
        if self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.GANLoss(prediction, target_tensor)
        elif self.gan_mode == 'vanilla':
            if target_is_real == 'gene':
                target_is_real = 'real'
            elif target_is_real == 'fake':
                target_is_real = 'gene'
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.GANLoss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def get_D_loss(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, 'real')
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, 'fake')
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    def get_rec_loss(self, outputs, targets):
        return self.criterionRec(input=outputs, target=targets)

    def backward_D(self):
        fake = self.fake_pool.query(self.fake.detach())
        self.loss_D = self.get_D_loss(self.netD, self.real, fake)
        return self.loss_D

    def backward_G(self):
        # print(f"fake {self.fake.shape}")
        self.loss_rec = self.get_rec_loss(self.rec_patches, self.label_patches)
        self.loss_gan = self.criterionGAN(self.netD(self.fake), 'gene') * self.lambda_gan
        self.loss_G = self.loss_rec + self.loss_gan 
        return self.loss_rec, self.loss_gan, self.loss_G
    
    def freeze_D_params(self):
        self.set_requires_grad([self.netD], False)

    def freeze_G_params(self):
        self.set_requires_grad([self.netG], False)

    def unfreeze_all_params(self):
        self.set_requires_grad([self.netD, self.netG], True)

    def set_input(self, images, bool_masked_pos, images_patch):
        self.inputs = images
        B, _, C = images_patch.shape
        self.label_patches = images_patch[bool_masked_pos].reshape(B, -1, C)
        self.mask = bool_masked_pos
        self.real = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
        self.real = rearrange(self.real, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=14, w=14)


    def forward(self):
        x_full, self.rec_patches = self.netG(self.inputs, self.mask)
        self.fake = rearrange(x_full, 'b n (p c) -> b n p c', c=3)
        self.fake = rearrange(self.fake, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=14, w=14)

        return self.rec_patches

class GANPretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 gan_mode='lsgan',
                 input_nc=3, 
                 ndf=64, 
                 netD='basic', 
                 n_layers_D=3,
                 pool_size=10000,
                 lambda_gan=0.0,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.netG = PretrainG(
            img_size=img_size, 
            patch_size=patch_size, 
            encoder_in_chans=encoder_in_chans, 
            encoder_num_classes=encoder_num_classes, 
            encoder_embed_dim=encoder_embed_dim, 
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads, 
            decoder_num_classes=decoder_num_classes, 
            decoder_embed_dim=decoder_embed_dim, 
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.netD = define_D(
            input_nc=input_nc, 
            ndf=ndf,
            netD=netD,
            n_layers_D=n_layers_D,
        )

        # gan set
        self.patch_size = patch_size
        target_real_label=1.0
        target_fake_label=-1.0
        target_gene_label=0.0
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('gene_label', torch.tensor(target_gene_label))
        self.criterionRec = nn.MSELoss()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.GANLoss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.GANLoss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.GANLoss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        self.fake_pool = ImagePool(pool_size)
        self.lambda_gan = lambda_gan

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real == 'real':
            target_tensor = self.real_label
        elif target_is_real == 'fake':
            target_tensor = self.fake_label
        elif target_is_real == 'gene':
            target_tensor = self.gene_label
        else:
            raise RuntimeError(f"target tensor wrong {target_is_real}")
        return target_tensor.expand_as(prediction)

    def criterionGAN(self, prediction, target_is_real):
        # print(f"pred {prediction.shape} {(prediction>0).float().mean()} {target_is_real}")
        if self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.GANLoss(prediction, target_tensor)
        elif self.gan_mode == 'vanilla':
            if target_is_real == 'gene':
                target_is_real = 'real'
            elif target_is_real == 'fake':
                target_is_real = 'gene'
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.GANLoss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real == 'real' or target_is_real == 'gene':
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def get_D_loss(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, 'real')
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, 'fake')
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    def get_rec_loss(self, outputs, targets):
        return self.criterionRec(input=outputs, target=targets)

    def backward_D(self):
        fake = self.fake_pool.query(self.fake.detach())
        self.loss_D = self.get_D_loss(self.netD, self.real, fake)
        return self.loss_D

    def backward_G(self):
        # print(f"fake {self.fake.shape}")
        self.loss_rec = self.get_rec_loss(self.rec_patches, self.label_patches)
        self.loss_gan = self.criterionGAN(self.netD(self.fake), 'gene') * self.lambda_gan
        self.loss_G = self.loss_rec + self.loss_gan 
        return self.loss_rec, self.loss_gan, self.loss_G
    
    def freeze_D_params(self):
        self.set_requires_grad([self.netD], False)

    def freeze_G_params(self):
        self.set_requires_grad([self.netG], False)

    def unfreeze_all_params(self):
        self.set_requires_grad([self.netD, self.netG], True)

    def set_input(self, images, bool_masked_pos, images_patch):
        self.inputs = images
        B, _, C = images_patch.shape
        self.label_patches = images_patch[bool_masked_pos].reshape(B, -1, C)
        self.mask = bool_masked_pos
        self.real = rearrange(images_patch, 'b n (p c) -> b n p c', c=3)
        self.real = rearrange(self.real, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=14, w=14)


    def forward(self, x, mask):
        return self.netG(x, mask)


@register_model
def pretrain_mae_small_patch16_224_tiny(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_mgan_small_patch16_224_wgangp_basic_SN(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        gan_mode='wgangp',
        input_nc=3, 
        ndf=64, 
        netD='basic_SN', 
        n_layers_D=3,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_mgan_small_patch16_224_lsgan_basic(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        gan_mode='lsgan',
        input_nc=3, 
        ndf=64, 
        netD='basic', 
        n_layers_D=3,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def test_pretrain_mgan_small_patch16_224_lsgan_basic(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        gan_mode='lsgan',
        input_nc=3, 
        ndf=64, 
        netD='basic', 
        n_layers_D=3,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_mae_base_patch16_224(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 

@register_model
def pretrain_mae_large_patch16_224(pretrained=False, **kwargs):
    model = GANPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model