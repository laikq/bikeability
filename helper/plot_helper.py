
def set_limits_from_data(ax, x, y, margin=0.05):
    xmarg = (max(x) - min(x)) * margin
    ymarg = (max(y) - min(y)) * margin
    ax.set_xlim(min(x) - xmarg, max(x) + xmarg)
    ax.set_ylim(min(y) - ymarg, max(y) + ymarg)
