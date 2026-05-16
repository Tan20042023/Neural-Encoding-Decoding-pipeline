
def keras_cc(obs, est):
    import tensorflow as tf
    """Pearson correlation coefficient"""
    x_mu = obs - tf.math.reduce_mean(obs, axis=0, keepdims=True)  # 使用 tf.math
    x_std = tf.math.reduce_std(obs, axis=0, keepdims=True)
    y_mu = est - tf.math.reduce_mean(est, axis=0, keepdims=True)
    y_std = tf.math.reduce_std(est, axis=0, keepdims=True)

    epsilon=tf.constant(1e-8, dtype=x_std.dtype)
    x_std = tf.math.maximum(x_std, epsilon)  # 使用 tf.math.maximum
    y_std = tf.math.maximum(y_std, epsilon)  # 使用 tf.math.maximum

    correlation = tf.math.reduce_mean((x_mu * y_mu), axis=0, keepdims=True) / (x_std * y_std)
    return tf.math.reduce_mean(correlation)  # 使用 tf.math.reduce_mean


def pytorch_cc(obs, est):
    import torch
    """纯PyTorch实现的Pearson相关系数"""
    x_mu = obs - torch.mean(obs, dim=0, keepdim=True)
    x_std = torch.std(obs, dim=0, keepdim=True, unbiased=True)
    y_mu = est - torch.mean(est, dim=0, keepdim=True)
    y_std = torch.std(est, dim=0, keepdim=True, unbiased=True)
    
    # 防止除零
    epsilon = 1e-8
    x_std = torch.clamp(x_std, min=epsilon)
    y_std = torch.clamp(y_std, min=epsilon)
    
    correlation = torch.mean((x_mu * y_mu), dim=0, keepdim=True) / (x_std * y_std)
    return torch.mean(correlation)