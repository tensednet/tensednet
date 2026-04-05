def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()
