def should_save_checkpoint(metric_value, threshold):
    """Return True only when the metric is strictly greater than the threshold."""
    return metric_value > threshold
