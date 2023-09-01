import matplotlib.pyplot as plt

# from https://github.com/alwynmathew/gradflow-check/blob/master/gradflow_check.py

def plot_gradient(params):
    layer_grads = []
    layer_names = []
    for n, p in params:
        if p.requires_grad and "bias" not in n:
            layer_names.append(n[:15])
            layer_grads.append(p.grad.abs().mean().cpu())
    
    plt.plot(layer_grads)
    plt.xticks(range(0, len(layer_grads), 1), layer_names, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(layer_grads))
    plt.xlabel("Layer Names")
    plt.ylabel("Average Layer Gradient")
    plt.grid(True)
    plt.show()