import torch
import torch.nn as nn

class HIVAE_NORM(nn.Module):
    def __init__(self):
        super(HIVAE_NORM, self).__init__()
        self.eps = 1e-6
        self.max = 1e20

    def _broadcast_mask(self, mask, static_types):
        new_mask = []
        for i, vtype in enumerate(static_types):
            if vtype[0] in ["cat", "ord"]:
                new_mask.append(
                    mask[:, i].unsqueeze(1).expand(-1, vtype[1]))
            else:
                new_mask.append(mask[:, i].unsqueeze(1))
        return torch.cat(new_mask, dim=1)

    def forward(self, x, mask, static_types, prior_parameters):

        mean_data = prior_parameters[0]
        std_data = prior_parameters[1]
        new_x = []

        for i, vtype in enumerate(static_types):
            x_i = torch.masked_select(x[..., i], mask[..., i].bool())
            new_x_i = torch.unsqueeze(x[..., i], -1)

            if vtype[0] in ["real", "truncate_norm"]:
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=self.eps, max=self.max)

                new_x_i = (new_x_i - mean_data[i]) / std_data[i]
            elif vtype[0] == "pos":
                x_i = torch.log1p(x_i)
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=self.eps, max=self.max)

                new_x_i = (torch.log1p(new_x_i) - mean_data[i]) / std_data[i]
            elif vtype[0] == "gamma":
                x_i = torch.log1p(x_i)
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=self.eps, max=self.max)

                new_x_i = (torch.log1p(new_x_i) - mean_data[i]) / std_data[i]
            elif vtype[0] == "count":
                new_x_i = torch.log1p(new_x_i)
            elif vtype[0] in ["cat", "ord"]:
                # pass
                # convert to one hot
                new_x_i = torch.nn.functional.one_hot(
                    new_x_i.long().squeeze(1), vtype[1])
            elif vtype[0] == "mse":
                new_x_i = new_x_i.float()

            if torch.isnan(new_x_i).any():
                raise ValueError(
                    f"NaN values found in normalized data for {vtype}"
                )
            if torch.isnan(mean_data[i]) or torch.isnan(std_data[i]):
                raise ValueError(
                    f"NaN values found in normalization parameters for {vtype}"
                )
            new_x.append(new_x_i)
        new_x = torch.cat(new_x, dim=-1)
        mask = self._broadcast_mask(mask, static_types)
        new_x = new_x * mask

        return (new_x, mask, (mean_data, std_data))

    def denormalize_params(self, x_params, static_types, norm_parameters):

        params = []
        for i, vtype in enumerate(static_types):
            mean_data = norm_parameters[0][i]
            std_data = norm_parameters[1][i]
            if vtype[0] in ["real", "truncate_norm"]:
                # mean and std
                mean = x_params[i][0] * std_data + mean_data
                std = x_params[i][1] * std_data
                params.append((mean, std))
            elif vtype[0] == "pos":
                # mean and std
                mean = x_params[i][0] * std_data + mean_data
                std = x_params[i][1] * std_data
                params.append((mean, std))
            elif vtype[0] == "gamma":
                # concentration and rate
                concentration  = x_params[i][0] * std_data + mean_data
                rate = x_params[i][1] / std_data
                params.append((concentration, rate))
            elif vtype[0] in ["count", "cat", "ord", "mse"]:
                params.append((x_params[i]))
            else:
                raise ValueError(f"Unknown data type {vtype[0]}")
        return params