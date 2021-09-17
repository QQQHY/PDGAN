import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#font = {'family' : 'normal',
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}
matplotlib.rc('font', **font)

if __name__ == '__main__': 
    workdir = 'ml-100k/'
    p3 = []
    p5 = []
    ndcg3 = []
    ndcg5 = []
    d3 = []
    d5 = []
    with open(workdir + 'gen_log_30.txt') as fin:
        for line in fin:
            line = line.split('\t')
            p3.append(line[1])
            p5.append(line[2])
            ndcg3.append(line[4])
            ndcg5.append(line[5])
            d3.append(line[7])
            d5.append(line[8])

    p3 = np.asarray(p3).astype(np.float32)
    p5 = np.asarray(p5).astype(np.float32)
    ndcg3 = np.asarray(ndcg3).astype(np.float32)
    ndcg5 = np.asarray(ndcg5).astype(np.float32)
    d3 = np.asarray(d3).astype(np.float32)
    d5 = np.asarray(d5).astype(np.float32)

    x = np.arange(300)

    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(x, p3, 'b--')
    # plt.xlabel('Training Epoch')
    # plt.ylabel('P@3')
    # plt.grid(True)
    # plt.subplot(212)
    # plt.plot(x, d3, 'r--')
    # plt.xlabel('Training Epoch')
    # plt.ylabel('CC@3')
    # plt.grid(True)
    # plt.show()
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    lns1 = ax1.plot(x, p3, '-', color=color, label='P')
    ax1.set_xlabel('Training Epoch', fontsize='18', weight='bold')
    ax1.set_ylabel('P@3', color=color, fontsize='18', weight='bold')
    ax1.tick_params('y', colors=color)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    color = 'tab:red'
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, d3, '--', color=color, label='CC')
    ax2.set_ylabel('CC@3', color=color, fontsize='18', weight='bold')
    ax2.tick_params('y', colors=color)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4, fontsize='14')

    fig.tight_layout()
    plt.title('PD-GAN', fontsize='18', weight='bold')
    plt.grid(True)
    plt.show()

    pass
