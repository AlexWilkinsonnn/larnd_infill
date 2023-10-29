import torch; import  torch.nn as nn
import MinkowskiEngine as ME


# NOTE not being used
class CompletionNet(nn.Module):

    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(
        self, pointcloud_size, in_nchannel=1, out_nchannel=1, final_pruning_threshold=None
    ):
        nn.Module.__init__(self)

        self.pointcloud_size = pointcloud_size
        self.final_pruning_threshold = final_pruning_threshold

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(in_nchannel, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6], dec_ch[5], kernel_size=4, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5], dec_ch[4], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4], dec_ch[3], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3], dec_ch[2], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2], dec_ch[1], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1], dec_ch[0], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], out_nchannel, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

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
            keep = keep + (t.F[:, 0] > self.final_pruning_threshold)

        keep = keep.squeeze()

        if not keep.shape and keep.item() or keep.sum().item() == 0:
            # print("keep.sum().item() == 0 in final pruning layer")
            return t

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
                target[curr_in[0].long()] = 1

        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in):
        enc_s1 = self.enc_block_s1(partial_in)

        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32

        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        # Remove voxels s32
        dec_s32 = self._pruning_layer(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16

        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        # Remove voxels s16
        dec_s16 = self._pruning_layer(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features

        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        # Remove voxels s16
        dec_s8 = self._pruning_layer(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        # print("dec_s8", dec_s8.shape)
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4

        dec_s4_cls = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        # Remove voxels s4
        dec_s4 = self._pruning_layer(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        # print("dec_s4", dec_s4.shape)
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2

        dec_s2_cls = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        # Remove voxels s2
        dec_s2 = self._pruning_layer(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        # dec_s1_cls = self.dec_s1_cls(dec_s1)

        # # Add encoder features
        dec_s1 = dec_s1 + enc_s1

        dec_s1_cls = self.dec_s1_cls(dec_s1)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        dec_s1_cls = self._pruning_layer(dec_s1_cls, keep_s1)
        dec_s1_cls = self._final_pruning_layer(dec_s1_cls)

        return dec_s1_cls


class CompletionNetSigMask(nn.Module):
    def __init__(
        self,
        pointcloud_size,
        in_nchannel=1, out_nchannel=1,
        final_pruning_threshold=None,
        final_layer="tanh",
        nonlinearity="elu",
        enc_ch=[16, 32, 64, 128, 256, 512, 1024],
        dec_ch=[16, 32, 64, 128, 256, 512, 1024]
    ):
        nn.Module.__init__(self)

        self.pointcloud_size = pointcloud_size
        self.final_pruning_threshold = final_pruning_threshold

        if nonlinearity == "elu":
            nonlinearity = ME.MinkowskiELU()
        elif nonlinearity == "relu":
            nonlinearity = ME.MinkowskiReLU()
        else:
            raise ValueError("nonlinearity: {} not valid".format(nonlinearity))

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_nchannel, enc_ch[0], kernel_size=3, stride=1, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, bias=True, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[0]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, bias=True, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[1]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[2]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, bias=True, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[3]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[4]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            # ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, bias=True, dimension=3),
            # ME.MinkowskiBatchNorm(enc_ch[5]),
            # ME.MinkowskiELU(),
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[6], dec_ch[5], kernel_size=4, stride=2, bias=True, dimension=3
        )
        self.dec_block_s32_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s32_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * dec_ch[5], dec_ch[5], kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s32s16_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[5], dec_ch[4], kernel_size=4, stride=2, bias=True, dimension=3,
        )
        self.dec_block_s16_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s16_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * dec_ch[4], dec_ch[4], kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s16s8_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[4], dec_ch[3], kernel_size=4, stride=2, bias=True, dimension=3,
        )
        self.dec_block_s8_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s8_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * dec_ch[3], dec_ch[3], kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s8s4_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[3], dec_ch[2], kernel_size=4, stride=2, bias=True, dimension=3,
        )
        self.dec_block_s4_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s4_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * dec_ch[2], dec_ch[2], kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s4s2_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[2], dec_ch[1], kernel_size=4, stride=2, bias=True, dimension=3,
        )
        self.dec_block_s2_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s2_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                2 * dec_ch[1], dec_ch[1], kernel_size=3, bias=True, dimension=3
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_block_s2s1_up = ME.MinkowskiConvolutionTranspose(
            enc_ch[1], dec_ch[0], kernel_size=4, stride=2, bias=True, dimension=3,
        )
        self.dec_block_s1_norm = nn.Sequential(
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )
        self.dec_block_s1_conv = nn.Sequential(
            ME.MinkowskiConvolution(2 * dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

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
        else:
            raise ValueError("final_layer: {} not valid".format(final_layer))

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
            keep = keep + (t.F[:, 0] > self.final_pruning_threshold)

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
                target[curr_in[0].long()] = 1

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
        dec_s32 = self.dec_block_s64s32_up(enc_s64, coordinates=enc_s32.coordinate_map_key)
        dec_s32 = self.dec_block_s32_norm(dec_s32)
        
        dec_s32 = ME.cat((dec_s32, enc_s32))

        dec_s32 = self.dec_block_s32_conv(dec_s32)

        ###################################################
        ## Decoder 32 -> 16
        ###################################################
        dec_s16 = self.dec_block_s32s16_up(dec_s32, coordinates=enc_s16.coordinate_map_key)
        dec_s16 = self.dec_block_s16_norm(dec_s16)
        
        dec_s16 = ME.cat((dec_s16, enc_s16))

        dec_s16 = self.dec_block_s16_conv(dec_s16)

        ###################################################
        ## Decoder 16 -> 8
        ###################################################
        dec_s8 = self.dec_block_s16s8_up(dec_s16, coordinates=enc_s8.coordinate_map_key)
        dec_s8 = self.dec_block_s8_norm(dec_s8)
        
        dec_s8 = ME.cat((dec_s8, enc_s8))

        dec_s8 = self.dec_block_s8_conv(dec_s8)

        ###################################################
        ## Decoder 8 -> 4
        ###################################################
        dec_s4 = self.dec_block_s8s4_up(dec_s8, coordinates=enc_s4.coordinate_map_key)
        dec_s4 = self.dec_block_s4_norm(dec_s4)
        
        dec_s4 = ME.cat((dec_s4, enc_s4))

        dec_s4 = self.dec_block_s4_conv(dec_s4)

        ###################################################
        ## Decoder 4 -> 2
        ###################################################
        dec_s2 = self.dec_block_s4s2_up(dec_s4, coordinates=enc_s2.coordinate_map_key)
        dec_s2 = self.dec_block_s2_norm(dec_s2)
        
        dec_s2 = ME.cat((dec_s2, enc_s2))

        dec_s2 = self.dec_block_s2_conv(dec_s2)

        ###################################################
        ## Decoder 2 -> 1
        ###################################################
        dec_s1 = self.dec_block_s2s1_up(dec_s2, coordinates=enc_s1.coordinate_map_key)
        dec_s1 = self.dec_block_s1_norm(dec_s1)
        
        dec_s1 = ME.cat((dec_s1, enc_s1))

        dec_s1 = self.dec_block_s1_conv(dec_s1)

        ###################################################
        ## Out
        ###################################################
        out = self.dec_out_conv(dec_s1)

        out = self._final_pruning_layer(out)

        out = self.final_layer(out)

        return out


# NOTE not being used
class MyCompletionNet(nn.Module):

    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(
        self, pointcloud_size, in_nchannel=1, out_nchannel=1, final_pruning_threshold=None
    ):
        nn.Module.__init__(self)

        self.pointcloud_size = pointcloud_size
        self.final_pruning_threshold = final_pruning_threshold

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(in_nchannel, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6], dec_ch[5], kernel_size=4, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], dec_ch[5], kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5], dec_ch[4], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], dec_ch[4], kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4], dec_ch[3], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], dec_ch[3], kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3], dec_ch[2], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], dec_ch[2], kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2], dec_ch[1], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], dec_ch[1], kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1], dec_ch[0], kernel_size=2, stride=2, dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], out_nchannel, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

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
            keep = keep + (t.F[:, 0] > self.final_pruning_threshold)

        keep = keep.squeeze()

        if not keep.shape and keep.item() or keep.sum().item() == 0:
            # print("keep.sum().item() == 0 in final pruning layer")
            return t

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
                target[curr_in[0].long()] = 1

        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in):
        enc_s1 = self.enc_block_s1(partial_in)

        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32

        dec_s32 = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32.F[:, -1] > 0).squeeze()

        # Remove voxels s32
        dec_s32 = self._pruning_layer(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16

        dec_s16 = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16.F[:, -1] > 0).squeeze()

        # Remove voxels s16
        dec_s16 = self._pruning_layer(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features

        dec_s8 = dec_s8 + enc_s8
        dec_s8 = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8.F[:, -1] > 0).squeeze()

        # Remove voxels s16
        dec_s8 = self._pruning_layer(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        # print("dec_s8", dec_s8.shape)
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4

        dec_s4 = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4.F[:, -1] > 0).squeeze()

        # Remove voxels s4
        dec_s4 = self._pruning_layer(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        # print("dec_s4", dec_s4.shape)
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2

        dec_s2 = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2.F[:, -1] > 0).squeeze()

        # Remove voxels s2
        dec_s2 = self._pruning_layer(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        # dec_s1_cls = self.dec_s1_cls(dec_s1)

        # # Add encoder features
        dec_s1 = dec_s1 + enc_s1

        dec_s1 = self.dec_s1_cls(dec_s1)
        keep_s1 = (dec_s1.F > 0).squeeze()

        dec_s1 = self._pruning_layer(dec_s1, keep_s1)
        dec_s1 = self._final_pruning_layer(dec_s1)

        return dec_s1

