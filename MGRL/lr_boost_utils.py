def build_boost_step_plan(boost_epoch):
    """Return the scheduler step index that precedes a boost epoch."""
    return boost_epoch - 1


def maybe_apply_lr_boost(optimizer, current_epoch, boost_epoch, boost_value, already_applied):
    """Apply a one-time LR boost at the configured epoch."""
    if already_applied:
        return True
    if boost_epoch is None or boost_value is None:
        return False
    if boost_epoch < 0:
        return False
    if current_epoch != boost_epoch:
        return False

    for group in optimizer.param_groups:
        group["lr"] = boost_value
    return True
