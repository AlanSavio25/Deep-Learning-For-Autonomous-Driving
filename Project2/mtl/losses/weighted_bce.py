import torch


class WeightedBCE(torch.nn.Module):
    @staticmethod
    def forward_one_image(prediction, label):

        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        cost = torch.nn.functional.binary_cross_entropy(prediction.float(), label.float(), weight=mask)

        if label.numel() == 0:
            return None, False
        return cost, True

    def forward(self, y_hat, y):
        assert torch.is_tensor(y_hat) and torch.is_tensor(y)
        assert y.dim() == 3 or y.dim() == 4 and y.shape[1] == 1
        if y.dim() == 4:
            y = y.squeeze(1)
        assert y_hat.dim() == 3 or y_hat.dim() == 4 and y_hat.shape[1] == 1
        if y_hat.dim() == 4:
            y_hat = y_hat.squeeze(1)
        assert y_hat.shape == y.shape
        N, H, W = y_hat.shape

        out_loss = torch.tensor(0, device=y.device, dtype=y.dtype, requires_grad=True)
        out_cnt = 0
        for i in range(N):
            loss, have_loss = self.forward_one_image(y_hat[i], y[i])
            if have_loss:
                out_loss = out_loss + loss
                out_cnt += 1

        return out_loss / max(out_cnt, 1)
