import torch
import torch.nn.functional as F
from mtl.datasets.definitions import MOD_DEPTH, MOD_SEMSEG, MOD_NORMAL

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, SelfAttention, DecoderNoSkipConnection

class ModelTaskDistillationWithNormal(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        num_classes_semseg = outputs_desc[MOD_SEMSEG]
        num_classes_depth = outputs_desc[MOD_DEPTH] # Is 1 since we predict a 1D continuous value, but could be changed to a classification problem
        num_classes_normal = outputs_desc[MOD_NORMAL]

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_normal = ASPP(ch_out_encoder_bottleneck, 256)

        self.first_decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_semseg)
        self.first_decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_depth)
        self.first_decoder_normal = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_normal)
        self.normalize_output_decoder_normal = torch.nn.Tanh() # Because the vector is normalized, hence cannot have values bigger than 1 or smaller than -1

        self.attention_semseg = SelfAttention(256, 256)
        self.attention_depth = SelfAttention(256, 256)
        self.attention_normal = SelfAttention(256, 256)

        self.second_decoder_semseg = DecoderNoSkipConnection(256, num_classes_semseg)
        self.second_decoder_depth = DecoderNoSkipConnection(256, num_classes_depth)


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_semseg = self.aspp_semseg(features_lowest)
        features_tasks_depth = self.aspp_depth(features_lowest)
        features_tasks_normal = self.aspp_normal(features_lowest)

        # Initial prediction after decoder #1, #2 and #3
        initial_predictions_semseg_4x, _, initial_penultimate_semseg_4x = self.first_decoder_semseg(features_tasks_semseg, features[4])
        initial_predictions_depth_4x, _, initial_penultimate_depth_4x = self.first_decoder_depth(features_tasks_depth, features[4])
        initial_predictions_normal_4x, _, initial_penultimate_normal_4x = self.first_decoder_normal(features_tasks_normal, features[4])

        initial_prediction_semseg_1x = F.interpolate(initial_predictions_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        initial_prediction_depth_1x = F.interpolate(initial_predictions_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)
        initial_prediction_normal_1x = F.interpolate(initial_predictions_normal_4x, size=input_resolution, mode='bilinear', align_corners=False)
        initial_normalized_normal_1x = self.normalize_output_decoder_normal(initial_prediction_normal_1x)

        after_attention_semseg = self.attention_semseg(initial_penultimate_semseg_4x)
        after_attention_depth = self.attention_depth(initial_penultimate_depth_4x)
        after_attention_normal = self.attention_normal(initial_penultimate_normal_4x)

        sum_semseg = torch.add(torch.add(initial_penultimate_semseg_4x, after_attention_depth), after_attention_normal)
        sum_depth = torch.add(torch.add(initial_penultimate_depth_4x, after_attention_semseg), after_attention_normal)

        # Final prediction after decoder #4 and #5
        final_prediction_semseg_4x = self.second_decoder_semseg(sum_semseg)
        final_prediction_depth_4x = self.second_decoder_depth(sum_depth)
        final_prediction_semseg_1x = F.interpolate(final_prediction_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        final_prediction_depth_1x = F.interpolate(final_prediction_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)


        out = {}

        out[MOD_SEMSEG] = [initial_prediction_semseg_1x, final_prediction_semseg_1x]
        out[MOD_DEPTH] = [initial_prediction_depth_1x, final_prediction_depth_1x]
        out[MOD_NORMAL] = initial_normalized_normal_1x

        return out
