import torch
import torch.nn.functional as F
from mtl.datasets.definitions import MOD_DEPTH, MOD_SEMSEG

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p

class ModelBranchedArchitecture(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        num_classes_semseg = outputs_desc[MOD_SEMSEG]
        num_classes_depth = outputs_desc[MOD_DEPTH] # Is 1 since we predict a 1D continuous value, but could be changed to a classification problem

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_semseg)
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_depth)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_semseg = self.aspp_semseg(features_lowest)
        features_tasks_depth = self.aspp_depth(features_lowest)

        predictions_semseg_4x, _, _ = self.decoder_semseg(features_tasks_semseg, features[4])
        predictions_depth_4x, _, _ = self.decoder_depth(features_tasks_depth, features[4])

        prediction_semseg_1x = F.interpolate(predictions_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        prediction_depth_1x = F.interpolate(predictions_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}

        out[MOD_SEMSEG] = prediction_semseg_1x
        out[MOD_DEPTH] = prediction_depth_1x

        return out
