import numpy as np
import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        # Flatten to 1D to avoid any multi-dimensional broadcast issues in masked_select
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.mean(torch.abs(true_masked - pred_masked))
    else:
        return torch.mean(torch.abs(true - pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.mean((pred_masked - true_masked) ** 2)
    else:
        return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.sqrt(torch.mean((pred_masked - true_masked) ** 2))
    else:
        return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.sqrt(torch.sum((pred_masked - true_masked) ** 2)) / torch.sqrt(torch.sum((true_masked - true_masked.mean()) ** 2))
    else:
        return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    # Input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape) == 4:
        # B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = pred * mask.float()
        true = true * mask.float()
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean) * (true - true_mean)).mean(dim=dims) / (pred_std * true_std)
    index = (true_std != 0)
    correlation = correlation[index].mean()
    return correlation

def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.mean(torch.abs(torch.div((true_masked - pred_masked), (true_masked + 0.001))))
    else:
        return torch.mean(torch.abs(torch.div((true - pred), (true + 0.001))))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        indicator = torch.gt(pred_masked - true_masked, 0).float()
        return indicator.mean()
    else:
        indicator = torch.gt(pred - true, 0).float()
        return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        bias = (true_masked + pred_masked) / (2 * true_masked)
        return bias.mean()
    else:
        bias = (true + pred) / (2 * true)
        return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.div(torch.sum(torch.abs((true_masked - pred_masked))), torch.sum(true_masked))
    else:
        return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        mask_flat = mask.reshape(-1)
        pred_masked = torch.masked_select(pred_flat, mask_flat)
        true_masked = torch.masked_select(true_flat, mask_flat)
        return torch.mean(torch.abs(true_masked - pred_masked) / (torch.abs(true_masked) + torch.abs(pred_masked)))
    else:
        return torch.mean(torch.abs(true - pred) / (torch.abs(true) + torch.abs(pred)))

def MAE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(pred - true))

def RMSE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.sqrt(np.mean(np.square(pred - true)))

def RRSE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred - true) ** 2)), np.sqrt(np.sum((true - mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true + 0.001))))

def PNBI_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred - true
    indicator = np.where(bias > 0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, axis=(0, 1))
        true = np.expand_dims(true, axis=(0, 1))
    elif len(pred.shape) == 3:
        pred = np.expand_dims(np.transpose(pred, (0, 2, 1)), axis=1)
        true = np.expand_dims(np.transpose(true, (0, 2, 1)), axis=1)
    elif len(pred.shape) == 4:
        pred = np.transpose(pred, (0, 1, 3, 2))
        true = np.transpose(true, (0, 1, 3, 2))
    else:
        raise ValueError
    dims = (0, 1, 2)
    if mask_value is not None:
        mask = np.where(true > mask_value, True, False)
        pred = pred * mask.astype(float)
        true = true * mask.astype(float)
    pred_mean = np.mean(pred, axis=dims)
    true_mean = np.mean(true, axis=dims)
    pred_std = np.std(pred, axis=dims)
    true_std = np.std(true, axis=dims)
    correlation = np.mean((pred - pred_mean) * (true - true_mean), axis=dims) / (pred_std * true_std)
    index = (true_std != 0)
    correlation = np.mean(correlation[index])
    return correlation

def All_Metrics(pred, true, mask1, mask2):
    assert type(pred) == type(true)
    if isinstance(pred, np.ndarray):
        mae = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        corr = CORR_np(pred, true, mask1)
    elif isinstance(pred, torch.Tensor):
        mae = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
    else:
        raise TypeError("Unsupported type for pred and true")
    return mae, rmse, mape, rrse, corr

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr

if __name__ == '__main__':
    pred = torch.Tensor([1, 2, 3, 4])
    true = torch.Tensor([2, 1, 4, 5])
    print(All_Metrics(pred, true, None, None))
