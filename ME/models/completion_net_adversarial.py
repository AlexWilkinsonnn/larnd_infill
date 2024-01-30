import os, functools
from collections import deque

import numpy as np

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from ME.losses import init_loss_func


class CompletionNetAdversarial(nn.Module):
    def __init__(self, conf):
        super(CompletionNetAdversarial, self).__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir
        self.adc_scalefactor = conf.scalefactors[0]

        self.net_G = CompletionNetSigMask(
            (conf.vmap["n_voxels"]["x"], conf.vmap["n_voxels"]["y"], conf.vmap["n_voxels"]["z"]),
            in_nchannel=conf.n_feats_in + 1, out_nchannel=conf.n_feats_out,
            final_pruning_threshold=conf.adc_threshold, **conf.model_params
        ).to(self.device)
        if conf.net_D == "MEClassifier":
            self.net_D = InfillDiscriminator(1, 1).to(self.device)
            raise NotImplementedError(
                "Need to implement prepping the prediction as a TensorField!"
            )
        elif conf.net_D == "PatchGAN":
            self.net_D = InfillDiscriminatorPatchGAN(1).to(self.device)
        else:
            raise NotImplementedError("no net_D '{}' defined".format(conf.net_D))

        if conf.optimizer_G == "SGD":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "Adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "RMSprop":
            self.optimizer_G = optim.RMSprop(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "AdamW":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "Adamax":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "NAdam":
            self.optimizer_G = optim.NAdam(self.net_G.parameters(), **conf.optimizer_G_params)
        elif conf.optimizer_G == "RAdam":
            self.optimizer_G = optim.RAdam(self.net_G.parameters(), **conf.optimizer_G_params)
        else:
            raise ValueError("{} not valid optimzer_G selection".format(conf.optimizer_G))
        if conf.optimizer_D == "SGD":
            self.optimizer_D = optim.SGD(self.net_D.parameters(), **conf.optimizer_D_params)
        else:
            raise ValueError("{} not valid optimzer_D selection".format(conf.optimizer_D))
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optimizer_G, 0.95)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optimizer_D, 0.95)

        self.lossfunc_G = init_loss_func(conf)
        if conf.D_type == "vanilla":
            self.lossfunc_D = nn.BCEWithLogitsLoss()
        elif conf.D_type == "lsgan":
            self.lossfunc_D = nn.MSELoss()
        else:
            raise ValueError("{} not valid D_type selection".format(conf.D_type))
        self.lambda_loss_GAN = getattr(conf, "loss_GAN_weight", 0.0)

        # pause D training if it is too strong
        if conf.D_training_stopper:
            self.recent_losses_G_GAN = deque(
                conf.D_training_stopper["window_len"] * [0],
                maxlen=conf.D_training_stopper["window_len"]
            )
            self.D_stop_threshold = conf.D_training_stopper["stop_loss_threshold"]
        else:
            self.recent_losses_G_GAN = None
        self.D_pause_until_epoch = conf.D_pause_until_epoch
        self.D_training_activated = False # activated in new_epoch
        self.stop_D_training = True

        self.register_buffer("real_label", torch.tensor(conf.real_label, device=self.device))
        self.register_buffer("fake_label", torch.tensor(conf.fake_label, device=self.device))

        self.loss_D_fake = None
        self.loss_D_real = None
        self.loss_D = None
        self.loss_G_GAN = None
        self.loss_G_pix_tot = None
        self.loss_G_comps = None
        self.loss_G = None

        self.data = None
        self.s_in = None
        self.s_target = None
        self.s_pred = None

    def set_input(self, data):
        # No images in the batch have anything to infill, not worth training on and breaks
        # loss calculations. Easier to skip at this stage than in the dataloader.
        if torch.all(data["input_feats"][:, -1] == 0):
            return False

        self.data = data
        self.s_in = ME.SparseTensor(
            coordinates=data["input_coords"], features=data["input_feats"], device=self.device
        )
        self.s_target = ME.SparseTensor(
            coordinates=data["target_coords"], features=data["target_feats"], device=self.device
        )

        return True

    def get_loss_names_weights(self):
        ret = self.lossfunc_G.get_names_scalefactors()
        ret["D_fake"] = 0.5
        ret["D_real"] = 0.5
        ret["D_tot"] = 1.0
        ret["G_GAN"] = self.lambda_loss_GAN
        ret["pixel_tot"] = 1.0
        return ret

    def get_current_losses(self, valid=False):
        losses_ret = self.loss_G_comps
        if not valid and self.loss_D is not None: # dont care how D does on unseen data
            losses_ret["D_fake"] = self.loss_D_fake
            losses_ret["D_real"] = self.loss_D_real
            losses_ret["D_tot"] = self.loss_D
        losses_ret["G_GAN"] = self.loss_G_GAN
        losses_ret["pixel_tot"] = self.loss_G_pix_tot
        losses_ret["tot"] = self.loss_G
        for loss_name, loss in losses_ret.items():
            losses_ret[loss_name] = loss.item()
        return losses_ret

    def get_current_visuals(self):
        return { "s_in" : self.s_in, "s_target" : self.s_target, "s_pred" : self.s_pred }

    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        self.net_G.eval()
        self.net_D.eval()

    def train(self):
        self.net_G.train()
        self.net_D.train()

    def scheduler_step(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def save_networks(self, suffix):
        torch.save(
            self.net_G.cpu().state_dict(),
            os.path.join(self.checkpoint_dir, "netG_{}.pth".format(suffix))
        )
        torch.save(
            self.net_D.cpu().state_dict(),
            os.path.join(self.checkpoint_dir, "netD_{}.pth".format(suffix))
        )
        self.net_G.to(self.device)
        self.net_D.to(self.device)

    def new_epoch(self, epoch):
        if not self.D_training_activated and epoch >= self.D_pause_until_epoch:
            self.D_training_activated = True
            self.stop_D_training = False

    def test(self, compute_losses=False):
        with torch.no_grad():
            self.forward()
            if compute_losses:
                # compute gan loss
                pred_fake = self.net_D(self.s_pred)
                self.loss_G_GAN = self.lossfunc_D(pred_fake, self.real_label.expand_as(pred_fake))

                # compute standard loss
                self.loss_G_pix_tot, self.loss_G_comps = self.lossfunc_G.calc_loss(
                    self.s_pred, self.s_in, self.s_target, self.data
                )

                # combine for final loss
                self.loss_G = (self.lambda_loss_GAN * self.loss_G_GAN + self.loss_G_pix_tot)

    def forward(self):
        self.s_pred = self.net_G(self.s_in)

    def _backward_D(self):
        s_in_infill_mask = self.s_in.F[:, -1] == 1
        s_in_infill_C = self.s_in.C[s_in_infill_mask]
        s_pred_infill_F = self.s_pred.features_at_coordinates(s_in_infill_C.type(torch.float))
        s_target_infill_F = self.s_target.features_at_coordinates(s_in_infill_C.type(torch.float))
        pred_tensor = ME.SparseTensor(
            coordinates=s_in_infill_C.detach(),
            features=s_pred_infill_F.detach(),
            device=self.device
        )
        target_tensor = ME.SparseTensor(
            coordinates=s_in_infill_C.detach(),
            features=s_target_infill_F.detach(),
            device=self.device
        )

        pred_fake = self.net_D(pred_tensor)
        self.loss_D_fake = self.lossfunc_D(pred_fake, self.fake_label.expand_as(pred_fake))

        target_real = self.net_D(target_tensor.detach())
        self.loss_D_real = self.lossfunc_D(target_real, self.real_label.expand_as(target_real))

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def _backward_G(self):
        # check what D makes of the output
        s_in_infill_mask = self.s_in.F[:, -1] == 1
        s_in_infill_C = self.s_in.C[s_in_infill_mask]
        s_pred_infill_F = self.s_pred.features_at_coordinates(s_in_infill_C.type(torch.float))
        pred_tensor = ME.SparseTensor(
            coordinates=s_in_infill_C, features=s_pred_infill_F, device=self.device
        )
        pred_fake = self.net_D(pred_tensor)
        self.loss_G_GAN = self.lossfunc_D(pred_fake, self.real_label.expand_as(pred_fake))

        # compute standard loss
        self.loss_G_pix_tot, self.loss_G_comps = self.lossfunc_G.calc_loss(
            self.s_pred, self.s_in, self.s_target, self.data
        )

        # combine for final loss
        self.loss_G = self.lambda_loss_GAN * self.loss_G_GAN + self.loss_G_pix_tot

        self.loss_G.backward()

        self._update_D_stopper()

    def _update_D_stopper(self):
        if not self.D_training_activated:
            return

        if self.recent_losses_G_GAN is not None:
            self.recent_losses_G_GAN.appendleft(self.loss_G_GAN.item())
            if np.mean(self.recent_losses_G_GAN) > self.D_stop_threshold:
                self.stop_D_training = True
            else:
                self.stop_D_training = False

    def _prep_D_input(self, s):
        """
        not being used anymore
        """
        C, F = s.C.detach(), s.F.detach()
        # Want to make sure D cannot use specific pixel values,
        # not sure if the architecture can do this but want to be sure
        F = (F * (1 / self.adc_scalefactor)).type(torch.int).type(torch.float) * self.adc_scalefactor

        if isinstance(self.net_D, InfillDiscriminator):
            # THIS IS IMPORTANT: no coordinates for a batch index causes avg pooling to create nan
            active_batches = torch.unique(C[:, 0])
            for i_batch in range(len(self.data["mask_x"])):
                if i_batch not in active_batches:
                    # print(active_batches)
                    # print("Adding {} and {}".format(torch.tensor([[i_batch, 0, 0, 0]], device=C.device), torch.tensor([[0.0]], device=C.device)))
                    C = torch.cat([C, torch.tensor([[i_batch, 0, 0, 0]], device=C.device)])
                    F = torch.cat([F, torch.tensor([[0.0]], device=C.device)])

            return ME.TensorField(coordinates=C.detach(), features=F.detach(), device=self.device)

    def optimize_parameters(self):
        # make prediction
        self.forward()

        # update D
        # if not self.stop_D_training:
        #     # print("updating D")
        #     self._set_requires_grad(self.net_D, True)
        #     self.optimizer_D.zero_grad()
        #     self._backward_D()
        #     self.optimizer_D.step()
        # else:
        #     # print("not updating D")
        #     pass

        # update G
        self._set_requires_grad(self.net_D, False)
        self.optimizer_G.zero_grad()
        self._backward_G()
        self.optimizer_G.step()


class CompletionNetSigMask(nn.Module):
    def __init__(
        self,
        pointcloud_size,
        in_nchannel=1, out_nchannel=1,
        final_pruning_threshold=None,
        final_layer="tanh",
        extra_convs=False,
        nonlinearity="elu",
        enc_ch=[16, 32, 64, 128, 256, 512, 1024],
        dec_ch=[16, 32, 64, 128, 256, 512, 1024]
    ):
        super(CompletionNetSigMask, self).__init__()

        self.pointcloud_size = pointcloud_size
        self.final_pruning_threshold = final_pruning_threshold

        if nonlinearity == "elu":
            nonlinearity = lambda: ME.MinkowskiELU() # just to be certain unique layers are used
        elif nonlinearity == "relu":
            nonlinearity = lambda: ME.MinkowskiReLU()
        else:
            raise ValueError("nonlinearity: {} not valid".format(nonlinearity))

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_nchannel, enc_ch[0], kernel_size=3, stride=1, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nonlinearity()
        )

        self.enc_block_s1s2 = self._make_encoder_block(
            enc_ch[0], enc_ch[1], nonlinearity, extra_convs
        )
        self.enc_block_s2s4 = self._make_encoder_block(
            enc_ch[1], enc_ch[2], nonlinearity, extra_convs
        )
        self.enc_block_s4s8 = self._make_encoder_block(
            enc_ch[2], enc_ch[3], nonlinearity, extra_convs
        )
        self.enc_block_s8s16 = self._make_encoder_block(
            enc_ch[3], enc_ch[4], nonlinearity, extra_convs
        )
        self.enc_block_s16s32 = self._make_encoder_block(
            enc_ch[4], enc_ch[5], nonlinearity, extra_convs
        )
        self.enc_block_s32s64 = self._make_encoder_block(
            enc_ch[5], enc_ch[6], nonlinearity, extra_convs
        )

        # Decoder
        (
            self.dec_block_s64s32_up,
            self.dec_block_s32_norm,
            self.dec_block_s32_post_cat_conv,
            self.dec_block_s64_conv
        ) = self._make_decoder_block(enc_ch[6], dec_ch[5], nonlinearity, extra_convs)
        (
            self.dec_block_s32s16_up,
            self.dec_block_s16_norm,
            self.dec_block_s16_post_cat_conv,
            self.dec_block_s32_conv
        ) = self._make_decoder_block(dec_ch[5], dec_ch[4], nonlinearity, extra_convs)
        (
            self.dec_block_s16s8_up,
            self.dec_block_s8_norm,
            self.dec_block_s8_post_cat_conv,
            self.dec_block_s16_conv
        ) = self._make_decoder_block(dec_ch[4], dec_ch[3], nonlinearity, extra_convs)
        (
            self.dec_block_s8s4_up,
            self.dec_block_s4_norm,
            self.dec_block_s4_post_cat_conv,
            self.dec_block_s8_conv
        ) = self._make_decoder_block(dec_ch[3], dec_ch[2], nonlinearity, extra_convs)
        (
            self.dec_block_s4s2_up,
            self.dec_block_s2_norm,
            self.dec_block_s2_post_cat_conv,
            self.dec_block_s4_conv
        ) = self._make_decoder_block(dec_ch[2], dec_ch[1], nonlinearity, extra_convs)
        (
            self.dec_block_s2s1_up,
            self.dec_block_s1_norm,
            self.dec_block_s1_post_cat_conv,
            self.dec_block_s2_conv
        ) = self._make_decoder_block(dec_ch[1], dec_ch[0], nonlinearity, extra_convs)

        self.dec_out_conv = ME.MinkowskiConvolution(
            dec_ch[0], out_nchannel, kernel_size=3, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

        # final layer
        if final_layer == "none":
            self.final_layer = lambda t: t
        elif final_layer == "tanh":
            self.final_layer = ME.MinkowskiTanh()
        elif final_layer == "hardtanh":
            self.final_layer = ME.MinkowskiHardtanh(min_val=0.0, max_val=1.0)
        else:
            raise ValueError("final_layer: {} not valid".format(final_layer))

    """ __init__ helpers """

    def _make_encoder_block(self, in_ch, out_ch, nonlinearity, extra_convs):
        enc_block= [
            ME.MinkowskiConvolution(
                in_ch, out_ch, kernel_size=2, stride=2, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(out_ch),
            nonlinearity(),
            ME.MinkowskiConvolution(
                out_ch, out_ch, kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(out_ch),
            nonlinearity()
        ]
        if extra_convs:
            conv_block = [
                ME.MinkowskiConvolution(
                    in_ch, in_ch, kernel_size=3, bias=True, dimension=3
                ),
                ME.MinkowskiBatchNorm(in_ch),
                nonlinearity()
            ]
            enc_block = conv_block + enc_block

        return nn.Sequential(*enc_block)

    def _make_decoder_block(self, in_ch, out_ch, nonlinearity, extra_convs):
        dec_up = ME.MinkowskiConvolutionTranspose(
            in_ch, out_ch, kernel_size=4, stride=2, bias=True, dimension=3
        )
        dec_post_up_norm = nn.Sequential(ME.MinkowskiBatchNorm(out_ch), nonlinearity())
        dec_post_cat_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * out_ch, out_ch, kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(out_ch),
            nonlinearity()
        )
        if extra_convs:
            dec_extra_conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_ch, in_ch, kernel_size=3, bias=True, dimension=3
                ),
                ME.MinkowskiBatchNorm(in_ch),
                nonlinearity()
            )
        else:
            dec_extra_conv = lambda t: t

        return dec_up, dec_post_up_norm, dec_post_cat_conv, dec_extra_conv

    """ end __init__ helpers """

    def _pruning_layer(self, t, keep):
        if keep.sum().item() == 0:
            return t

        out = self.pruning(t, keep)

        return out

    def _final_pruning_layer(self, t):
        """Remove coords outside of active volume"""
        keep = (
            (t.C[:, 1] < self.pointcloud_size[0]) *
            (t.C[:, 2] < self.pointcloud_size[1]) *
            (t.C[:, 3] < self.pointcloud_size[2])
        )
        if self.final_pruning_threshold is not None:
            keep = keep * (t.F[:, 0] > self.final_pruning_threshold)

        keep = keep.squeeze()

        try:
            if not keep.shape and keep.item() or keep.sum().item() == 0:
                return t
        except:
            print(keep)
            print(keep.shape)
            raise Exception

        try:
            out = self.pruning(t, keep)
        except RuntimeError as e:
            print(keep)
            print(keep.shape)
            raise e

        return out

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(target_key, out.tensor_stride[0])
            kernel_map = cm.kernel_map(
                out.coordinate_map_key, strided_target_key, kernel_size=kernel_size, region_type=1
            )
            for _, curr_in in kernel_map.items():
                arget[curr_in[0].long()] = 1

        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, input_t):
        enc_s1 = self.enc_block_s1(input_t)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ###################################################
        ## Decoder 64 -> 32
        ###################################################
        dec_s64 = self.dec_block_s64_conv(enc_s64)

        dec_s32 = self.dec_block_s64s32_up(dec_s64, coordinates=enc_s32.coordinate_map_key)
        dec_s32 = self.dec_block_s32_norm(dec_s32)

        dec_s32 = ME.cat((dec_s32, enc_s32))
        dec_s32 = self.dec_block_s32_post_cat_conv(dec_s32)

        ###################################################
        ## Decoder 32 -> 16
        ###################################################
        dec_s32 = self.dec_block_s32_conv(enc_s32)

        dec_s16 = self.dec_block_s32s16_up(dec_s32, coordinates=enc_s16.coordinate_map_key)
        dec_s16 = self.dec_block_s16_norm(dec_s16)

        dec_s16 = ME.cat((dec_s16, enc_s16))
        dec_s16 = self.dec_block_s16_post_cat_conv(dec_s16)

        ###################################################
        ## Decoder 16 -> 8
        ###################################################
        dec_s16 = self.dec_block_s16_conv(enc_s16)

        dec_s8 = self.dec_block_s16s8_up(dec_s16, coordinates=enc_s8.coordinate_map_key)
        dec_s8 = self.dec_block_s8_norm(dec_s8)

        dec_s8 = ME.cat((dec_s8, enc_s8))
        dec_s8 = self.dec_block_s8_post_cat_conv(dec_s8)

        ###################################################
        ## Decoder 8 -> 4
        ###################################################
        dec_s8 = self.dec_block_s8_conv(enc_s8)

        dec_s4 = self.dec_block_s8s4_up(dec_s8, coordinates=enc_s4.coordinate_map_key)
        dec_s4 = self.dec_block_s4_norm(dec_s4)

        dec_s4 = ME.cat((dec_s4, enc_s4))
        dec_s4 = self.dec_block_s4_post_cat_conv(dec_s4)

        ###################################################
        ## Decoder 4 -> 2
        ###################################################
        dec_s4 = self.dec_block_s4_conv(enc_s4)

        dec_s2 = self.dec_block_s4s2_up(dec_s4, coordinates=enc_s2.coordinate_map_key)
        dec_s2 = self.dec_block_s2_norm(dec_s2)

        dec_s2 = ME.cat((dec_s2, enc_s2))
        dec_s2 = self.dec_block_s2_post_cat_conv(dec_s2)

        ###################################################
        ## Decoder 2 -> 1
        ###################################################
        dec_s2 = self.dec_block_s2_conv(enc_s2)

        dec_s1 = self.dec_block_s2s1_up(dec_s2, coordinates=enc_s1.coordinate_map_key)
        dec_s1 = self.dec_block_s1_norm(dec_s1)

        dec_s1 = ME.cat((dec_s1, enc_s1))
        dec_s1 = self.dec_block_s1_post_cat_conv(dec_s1)

        ###################################################
        ## Out
        ###################################################
        out = self.dec_out_conv(dec_s1)

        out = self.final_layer(out)

        out = self._final_pruning_layer(out)

        return out

class InfillDiscriminatorPatchGAN(nn.Module):
    """
    Aim to reproduce pix2pix style PatchGAN. Sparse images will have different numbers of patches
    depending on the number of coordinates but the patches will be spatially the same.
    The PatchGAN expects as input infill coordinates + some amount of unmasked coordinates from
    either side of the infill (so it knows the condition but isnt just classifying patches that are
    majority unmasked and so easy for the generator to reproduce)
    """
    def __init__(
        self,
        in_channel,
        n_layers=2,
        nf=64,
        norm_layer=ME.MinkowskiInstanceNorm,
        dim=3
    ):
        super(InfillDiscriminatorPatchGAN, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Downsampling convolutional net
        seq = [
            ME.MinkowskiConvolution(in_channel, nf, kernel_size=4, stride=2, dimension=dim),
            ME.MinkowskiLeakyReLU(0.2, True)
        ]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            seq += [
                ME.MinkowskiConvolution(
                    nf * nf_mult_prev, nf * nf_mult,
                    kernel_size=4, stride=2, bias=use_bias, dimension=dim
                ),
                norm_layer(nf * nf_mult),
                ME.MinkowskiLeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        seq += [
            ME.MinkowskiConvolution(
                nf * nf_mult_prev, nf * nf_mult,
                kernel_size=4, stride=1, bias=use_bias, dimension=dim
            ),
            norm_layer(nf * nf_mult),
            ME.MinkowskiLeakyReLU(0.2, True)
        ]

        # 1 channel prediction for each patch
        seq += [ME.MinkowskiConvolution(nf * nf_mult, 1, kernel_size=4, stride=1, dimension=dim)]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        patch_preds = self.model(x)
        # Take the minimum of each batch's patch predictions to get real/fake prediction for batch
        # ie. global min pooling
        # My thought is D should give high scores to patches that look real and low to ones that
        # look fake. Most of the image comes from unmasked track so it is trivial for G to make
        # this look real. Image should be classified as fake based on the fakest looking patch D
        # sees (this should correspond to the worst infill)
        # Don't want to use average pool because that will depend on the number of patches of
        # unmasked track which varies a lot
        batch_preds = torch.cat(
            [
                torch.mean(patch_preds.F[patch_preds.C[:, 0] == i_b]).view(1, 1)
                for i_b in torch.unique(patch_preds.C[:, 0])
            ]
        )
        return batch_preds

class InfillDiscriminator(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3
    ):
        super(InfillDiscriminator, self).__init__()

        self.D = D

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = self.get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv4 = self.get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x : ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        # print("x2")
        # print(x2)
        # print(torch.isnan(x2.F).sum(), x2.F.size())
        # print("y")
        # print(y)
        # print(torch.isnan(y.F).sum(), y.F.size())
        # print(torch.unique(x.C[:, 0]))
        # print("=======")

        return self.final(ME.cat(x1, x2)).F

