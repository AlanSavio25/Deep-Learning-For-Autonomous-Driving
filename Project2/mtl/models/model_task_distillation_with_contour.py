import torch
import torch.nn.functional as F
from mtl.datasets.definitions import MOD_DEPTH, MOD_SEMSEG, MOD_CONTOUR

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, SelfAttention, DecoderNoSkipConnection

class ModelTaskDistillationWithContour(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        num_classes_semseg = outputs_desc[MOD_SEMSEG]
        num_classes_depth = outputs_desc[MOD_DEPTH] # Is 1 since we predict a 1D continuous value, but could be changed to a classification problem
        num_classes_contour = outputs_desc[MOD_CONTOUR]

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_contour = ASPP(ch_out_encoder_bottleneck, 256)

        self.first_decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_semseg)
        self.first_decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_depth)
        self.first_decoder_contour = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_classes_contour)

        self.attention_semseg = SelfAttention(256, 256) # What should be outchannels?
        self.attention_depth = SelfAttention(256, 256) # What should be outchannels?
        self.attention_contour = SelfAttention(256, 256)

        self.second_decoder_semseg = DecoderNoSkipConnection(256, num_classes_semseg) # Should take input one layer before where 256 imput channels
        self.second_decoder_depth = DecoderNoSkipConnection(256, num_classes_depth) # Should take input one layer before where 256 imput channels


    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_semseg = self.aspp_semseg(features_lowest)
        features_tasks_depth = self.aspp_depth(features_lowest)
        features_tasks_contour = self.aspp_contour(features_lowest)

        # Initial prediction after decoder #1 and #2
        initial_predictions_semseg_4x, _, initial_penultimate_semseg_4x = self.first_decoder_semseg(features_tasks_semseg, features[4])
        initial_predictions_depth_4x, _, initial_penultimate_depth_4x = self.first_decoder_depth(features_tasks_depth, features[4])
        initial_predictions_contour_4x, _, initial_penultimate_contour_4x = self.first_decoder_contour(features_tasks_contour, features[4])

        initial_prediction_semseg_1x = F.interpolate(initial_predictions_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        initial_prediction_depth_1x = F.interpolate(initial_predictions_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)
        initial_prediction_contour_1x = F.interpolate(initial_predictions_contour_4x, size=input_resolution, mode='bilinear', align_corners=False)

        after_attention_semseg = self.attention_semseg(initial_penultimate_semseg_4x)
        after_attention_depth = self.attention_depth(initial_penultimate_depth_4x)
        after_attention_contour = self.attention_contour(initial_penultimate_contour_4x)

        sum_semseg = torch.add(torch.add(initial_penultimate_semseg_4x, after_attention_depth), after_attention_contour)
        sum_depth = torch.add(torch.add(initial_penultimate_depth_4x, after_attention_semseg), after_attention_contour)

        # Final prediction after decoder #3 and #4
        final_prediction_semseg_4x = self.second_decoder_semseg(sum_semseg)
        final_prediction_depth_4x = self.second_decoder_depth(sum_depth)
        final_prediction_semseg_1x = F.interpolate(final_prediction_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        final_prediction_depth_1x = F.interpolate(final_prediction_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)


        out = {}

        out[MOD_SEMSEG] = [initial_prediction_semseg_1x, final_prediction_semseg_1x]
        out[MOD_DEPTH] = [initial_prediction_depth_1x, final_prediction_depth_1x]
        out[MOD_CONTOUR] = initial_prediction_contour_1x

        return out
