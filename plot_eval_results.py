import re
from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt


table1 = """name	steps (k)	tokens (B)	loss	arc	mmlu
MoE	0	0	12.18	27.73	25.14
MoE	20	50	1.9	37.88	26.71
MoE	22	83	1.807	38.99	27.76
MoE	23	96.469	1.792	39.76	27.36
MoE data-v2	7	122.16	1.948	38.48	
MoE data-v2	8	125.83	1.862	38.48	25.73
MoE data-v2	11	136.84	1.858	37.97	26.83
MoE data-v2	13	144.18	1.848	38.74	26.77
MoE data-v2	14	147.85	1.875	37.97	26.99
MoE data-v2	16	155.189	1.883	37.88	25.44
MoE data-v2	17	158.86	1.896	37.71	27.03
RandMoE	0	0	11.25	29.1	24.46
RandMoE	2	8.4	2.87	22.7	25.6
RandMoE	5	12.58	2.579	22.87	24.51
RandMoE	8	33.55	2.169	25.34	24.76
GateLossλ0.1	6	22.02	1.991	33.53	
GateLossλ0.1	7	25.69	2.022	35.49	27.02
GateLossλ0.1	8	29.36	1.941	35.92	26.4
GateLossλ0.1	10	36.7	1.94	36.43	27.62
GateLossλ0.1	12	44.04	1.922	37.29	26.28
GateLossλ0.1	13	47.71	1.844	37.54	27.05
GateLossλ0.1	15	55.05	1.833	37.71	26.62
GateLossλ0.1	16	58.72	1.864	37.54	26.77
GateLossλ0.1	17	62.39	1.852	38.91	27.44
GateLossλ0.1	18	66.06	1.833	39.76	27.28
GateLossλ0.1	25	91.75	1.825	39.42	27.34
GateLossλ0.1	26	95.42	1.81	39.85	27.48
GateLossλ0.1	27	99.09	1.81	40.27	27.44"""

table2 = """name	steps (k)	tokens (B)	loss	arc	mmlu
MoE	20	50	1.9	37.88	26.71
MoE	22	83	1.807	38.99	27.76
MoE	23	96.469	1.792	39.76	27.36
MoE data-v2	7	122.16	1.948	38.48	
MoE data-v2	8	125.83	1.862	38.48	25.73
MoE data-v2	11	136.84	1.858	37.97	26.83
MoE data-v2	13	144.18	1.848	38.74	26.77
MoE data-v2	14	147.85	1.875	37.97	26.99
MoE data-v2	16	155.189	1.883	37.88	25.44
MoE data-v2	17	158.86	1.896	37.71	27.03
RandMoE	2	8.4	2.87	22.7	25.6
RandMoE	5	12.58	2.579	22.87	24.51
RandMoE	8	33.55	2.169	25.34	24.76
GateLossλ0.1	6	22.02	1.991	33.53	
GateLossλ0.1	7	25.69	2.022	35.49	27.02
GateLossλ0.1	8	29.36	1.941	35.92	26.4
GateLossλ0.1	10	36.7	1.94	36.43	27.62
GateLossλ0.1	12	44.04	1.922	37.29	26.28
GateLossλ0.1	13	47.71	1.844	37.54	27.05
GateLossλ0.1	15	55.05	1.833	37.71	26.62
GateLossλ0.1	16	58.72	1.864	37.54	26.77
GateLossλ0.1	17	62.39	1.852	38.91	27.44
GateLossλ0.1	18	66.06	1.833	39.76	27.28
GateLossλ0.1	25	91.75	1.825	39.42	27.34
GateLossλ0.1	26	95.42	1.81	39.85	27.48
GateLossλ0.1	27	99.09	1.81	40.27	27.44"""

x_label = "loss"  # token or loss
benchmark = "arc"  # arc or mmlu
markers = cycle("ox+*")

assert x_label in ["token", "loss"]
assert benchmark in ["arc", "mmlu"]

label_to_data = defaultdict(lambda: {"xs": [], "ys": []})
xs = []
ys = []
for line in table2.split("\n"):
    name, steps, tokens, loss, arc, mmlu = line.split("\t")
    if name == "name":
        continue

    y = None
    if benchmark == "arc" and re.match(r"[\d\.]+", arc):
        y = float(arc)
    elif benchmark == "mmlu" and re.match(r"[\d\.]+", mmlu):
        y = float(mmlu)

    x = None
    if x_label == "token":
        x = float(tokens)
    elif x_label == "loss":
        x = float(loss)

    if x and y:
        label_to_data[name]["xs"].append(x)
        label_to_data[name]["ys"].append(y)

fig = plt.figure()
ax = fig.add_subplot(111)
for name in label_to_data:
    xs = label_to_data[name]["xs"]
    ys = label_to_data[name]["ys"]
    ax.plot(xs, ys, label=name, marker=next(markers))
ax.legend()
ax.set_title(f"{benchmark.upper()} on {x_label}")
ax.set_xlabel(x_label)
ax.grid(True)
ax.set_axisbelow(True)
fig.savefig(f"results/figs/{x_label}_{benchmark}.png")
plt.close()
