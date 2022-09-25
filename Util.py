import itertools
import os
import json
import shutil

import matplotlib.colors
import Population as Pop
import numpy as np
from scipy import stats as st
import pickle
import pandas as pd
import csv
from collections import Counter
from scipy.stats import linregress
import matplotlib.pyplot as plt
import re
import Constants as C
import matplotlib.patches as mpatches
from PIL import Image
import cv2
import os
import xlsxwriter
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compile_simulation_data(dir):
    existing_sims = []
    sim_data = os.path.join(dir, "simulation_data.csv")
    if os.path.exists(sim_data):
        print(f"\rSimulation data exists. Adding to existing data...", end="\t")
        df = pd.read_csv(sim_data)
        if df[df["name"] == "count"].first_valid_index():
            print(f"\rPrevious summation data found. Removing summation data from: {os.path.split(dir)[1]}",  end="\t")
            first_id = df[df["name"] == "count"].first_valid_index()
            df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
        df.dropna(inplace=True)
        existing_sims = list(df["name"])
        df.to_csv(sim_data, index=False)
    else:
        print(f"\rCreating new simulation data file...", end="\t")
    print(f"\rCompiling simulation data for {dir}", end="")
    compiled_data = []
    for dirs, subdirs, files in os.walk(dir):
        if os.path.split(dirs)[1] not in existing_sims and os.path.split(os.path.split(dirs)[0])[1].startswith("t"):
            data_dict = {}
            data_dict["name"] = os.path.basename(os.path.normpath(dirs))
            neuron_fp = os.path.join(dirs, "neuron_data.json")
            if os.path.exists(neuron_fp):
                with open(neuron_fp, "r") as file:
                    data = json.load(file)
                    for id in data:
                        spikes = len(data[id]["spikes"])
                        spike_rate = spikes / (data[id]["duration"] / 1000)
                        type = "i" if data[id]["type"] == "<class 'Population.Input'>" else "n"
                        data_dict[f"{type}{id}_SR"] = spike_rate
            synapse_fp = os.path.join(dirs, "synapse_data.json")
            if os.path.exists(synapse_fp):
                with open(synapse_fp, "r") as file:
                    data = json.load(file)
                    keys = data.keys()
                    for k in keys:
                        try:
                            l = data[k]["d_hist"]["d"]
                        except:
                            l = data[k]["d_hist"]
                        saturation = check_saturation(l, C.SATURATION_LENGTH)
                        if data_dict[f"n0_SR"] <= 0.1:
                            data_dict[k] = "dormant"
                        elif saturation:
                            data_dict[k] = saturation
                        else:
                            if check_convergence(l, C.CONVERGENT_LENGTH):
                                if k in data_dict.keys():
                                    data_dict[k] += "-converging"
                                else:
                                    data_dict[k] = "converging"
                            elif check_repetitiveness(l, C.REPETITIVE_LENGTH):
                                if k in data_dict.keys():
                                    data_dict[k] += "-repeating"
                                else:
                                    data_dict[k] = "repeating"
                            try:
                                divergence = check_divergence(data[k]["d_hist"]["d"])
                            except:
                                divergence = check_divergence(data[k]["d_hist"])
                            if divergence:
                                data_dict[k] = divergence
                        if k not in data_dict.keys():
                            data_dict[k] = "uncategorized"
            compiled_data.append(data_dict)
    save_sim_data(dir, compiled_data)
    print("Finished compiling data for: ", dir)


def save_sim_data(dir, data):
    path = os.path.join(dir, "simulation_data.csv")
    exists = os.path.isfile(path)
    with open(path, 'a' if exists else 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(data)


def check_repetitiveness(l, pattern_length, use_correlation=False):
    if use_correlation:
        pattern = l[-pattern_length:]
        string = l[-pattern_length * 4: -pattern_length]
        max_corr = 0
        for i in range(len(string) - len(pattern)):
            corr = np.corrcoef(string[i:i + len(pattern)], pattern)[0][1]
            max_corr = max(max_corr, corr)
        if max_corr > C.CORRELATION_THRESHOLD:
            return True
        else:
            return False
    else:
        pattern = str(l[-pattern_length:]).strip("[]")
        string = str(l[: -pattern_length]).strip("[]")
        if pattern in string:
            return True
        else:
            return False


def check_convergence(l, pattern_length):
    if np.std(l[-pattern_length:]) < C.STD_THRESHOLD:
        return True
    else:
        return False


def check_divergence(l):
    slope = linregress(np.round(np.arange(0, np.round(len(l)/10, 1), 0.1), 1), l)[0]
    if slope > C.SLOPE_THRESHOLD:
        return "increasing"
    elif slope < -C.SLOPE_THRESHOLD:
        return "decreasing"
    else:
        return False


def check_saturation(l, pattern_length):
    mean_val = np.mean(l[-pattern_length:])
    if C.MIN_DELAY - C.STD_THRESHOLD < mean_val < C.MIN_DELAY + C.STD_THRESHOLD:
        return "min"
    elif C.MAX_DELAY - C.STD_THRESHOLD < mean_val < C.MAX_DELAY + C.STD_THRESHOLD:
        return "max"
    else:
        return False


def get_SR_data(dir):
    spike_rates = []
    reservoir_spike_rates = []
    input_spike_rates = []
    dormant = 0
    reservoir = 0
    inputs = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file == "neuron_data.json":
                print(f"\rCompiling SR data for: {os.path.split(dirs)[1]}", end="")
                with open(os.path.join(subdir, file), "r") as file:
                    data = json.loads(file.read())
                    for id in data:
                        spikes = len(data[id]["spikes"])
                        if spikes == 0:
                            dormant += 1
                        spike_rate = spikes / (data[id]["duration"] / 1000)
                        spike_rates.append(spike_rate)
                        if data[id]["type"] == str(Pop.Input):
                            input_spike_rates.append(spike_rate)
                            inputs += 1
                        else:
                            reservoir_spike_rates.append(spike_rate)
                            reservoir += 1

    avg_sr = float(np.round(np.mean(spike_rates), 1))
    median_sr = float(np.round(np.median(spike_rates), 1))
    mode_sr = np.round(list(st.mode(spike_rates)[0]), 1).tolist()
    std_sr = float(np.round(np.std(spike_rates), 1))

    avg_rsr = float(np.round(np.mean(reservoir_spike_rates), 1))
    median_rsr = float(np.round(np.median(reservoir_spike_rates), 1))
    mode_rsr = np.round(list(st.mode(reservoir_spike_rates)[0]), 1).tolist()
    std_rsr = float(np.round(np.std(reservoir_spike_rates), 1))

    avg_isr = float(np.round(np.mean(input_spike_rates), 1))
    median_isr = float(np.round(np.median(input_spike_rates), 1))
    mode_isr = np.round(st.mode(input_spike_rates)[0], 1).tolist()
    std_isr = float(np.round(np.std(input_spike_rates), 1))

    dormant_rate = float(np.round(dormant / reservoir if reservoir != 0 else 0, 2))
    SR_data = {
        "inputs": inputs,
        "reservoir": reservoir,
        "dormant neurons": dormant,
        "dormant rate": dormant_rate,
        "average SR": avg_sr,
        "median SR": median_sr,
        "mode SR": mode_sr,
        "std SR": std_sr,
        "average reservoir SR": avg_rsr,
        "median reservoir SR": median_rsr,
        "mode reservoir SR": mode_rsr,
        "std reservoir SR": std_rsr,
        "average input SR": avg_isr,
        "median input SR": median_isr,
        "mode input SR": mode_isr,
        "std input SR": std_isr
    }
    with open(f"{dir}/SR_data.json", "w") as f:
        json.dump(SR_data, f)


def sum_simulation_data(dir):
    print(f"\rSumming simulation data...", end="")
    df = pd.read_csv(os.path.join(dir, "simulation_data.csv"))
    if df[df["name"] == "count"].first_valid_index():
        print(f"\rDeleting previous summation data...", end="")
        first_id = df[df["name"] == "count"].first_valid_index()
        df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
        print(f"\rSumming simulation data...", end="")
    df.dropna(inplace=True)
    count = {"name": "count"}
    mean = {"name": "mean"}
    std = {"name": "std"}
    min = {"name": "min"}
    pros25 = {"name": "25%"}
    pros50 = {"name": "50%"}
    pros75 = {"name": "75%"}
    max = {"name": "max"}
    dormant = {"name": "dormant"}
    cols = list(df.columns)
    for col in cols:
        if not col == "name":
            if df[col].dtypes == object:
                counts = Counter(list(df[col]))
                count[col] = dict(counts)
                mean[col] = counts.most_common()[0]
            else:
                stats = round(df[col].describe(), 1)
                count[col] = stats["count"]
                mean[col] = stats["mean"]
                std[col] = stats["std"]
                min[col] = stats["min"]
                pros25[col] = stats["25%"]
                pros50[col] = stats["50%"]
                pros75[col] = stats["75%"]
                max[col] = stats["max"]
                dormant[col] = (df[col] == 0).sum()
    df.to_csv(os.path.join(dir, "simulation_data.csv"), index=False)
    data_list = [count, mean, std, min, pros25, pros50, pros75, max, dormant]
    with open(os.path.join(dir, "simulation_data.csv"), 'a', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cols)
        for d in data_list:
            writer.writerow(d)
    print("Finished summing data for: ", dir)

def delete_simulation_data(dir):
    if os.path.exists(os.path.join(dir, "simulation_data.csv")):
        os.remove(os.path.join(dir, "simulation_data.csv"))
        print("DELETED simulation data for: ", os.path.split(os.path.split(dir)[0])[1])
    else:
        print("No simulation data found for: ", os.path.split(os.path.split(dir)[0])[1])


def save_model(object, dir):
    with open(dir, "wb") as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)


def load_model(dir):
    print("Loading: ", dir)
    with open(os.path.join(dir), 'rb') as file:
        model = pickle.load(file)
        new_dir = os.path.split(dir)[0]
        model.dir = new_dir
        return model


def plot_delay_categories(dir, file_title, identifier_title, identifiers, nd):
    cat_list = C.DELAY_CATEGORIES_SHORTLIST
    possible_colors = C.DCAT_COLORS[:len(cat_list)]
    for dirs, subdirs, files in os.walk(dir):
        if os.path.exists(os.path.join(dirs, "simulation_data.csv")):
            df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
            index = list(df.index[df["name"] == "count"])[0]
            df_rows = df.loc[:index - 1]
            data = [[] for x in range(df_rows.shape[0])]
            for col in df_rows.keys():
                if col == "name":
                    names = list(df_rows[col])
                    for i, name in enumerate(names):
                        f = []
                        d = []
                        for fx in identifiers:
                            f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
                        for dx in range(nd):
                            d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
                        config = (f + d)
                        [data[i].append(x) for x in config]
                elif re.match("[0-9]+_[0-9]+$", col):
                    conn = list(df_rows[col])
                    for i, c in enumerate(conn):
                        data[i].append(c)
            data = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                                       range(len(identifiers))], 2)]),
            np.mean([abs(x[0] - x[1]) for x in
                     itertools.combinations([element[x] for x in
                                             range(len(identifiers), len(identifiers) + nd)], 2)]),
            element[0], element[len(identifiers)]))
            x = []
            y = []
            z = []
            for row in data:
                x_str = ""
                y_str = ""
                cat = []
                for yid in range(len(identifiers)):
                    if yid < len(identifiers) - 1:
                        y_str += f"{row[yid]}-"
                    else:
                        y_str += f"{row[yid]}"
                for xid in range(len(identifiers), nd + len(identifiers)):
                    if xid < nd + len(identifiers) - 1:
                        x_str += f"{row[xid]}-"
                    else:
                        x_str += f"{row[xid]}"
                x.append(x_str)
                y.append(y_str)
                for catid in range(nd + len(identifiers), len(row)):
                    cat.append(row[catid])
                if any(i in cat for i in ["increasing", "decreasing", "min", "max"]):
                    z.append("diverging")
                elif "uncategorized" in cat:
                    z.append("uncategorized")
                elif "repeating" in cat:
                    z.append("repeating")
                elif "converging" in cat:
                    z.append("converging")
                else:
                    raise Exception(f"Category {cat} not found!")
            fig, ax = plt.subplots()
            colors = []
            for p in z:
                if p in cat_list:
                    index = cat_list.index(p)
                else:
                    raise Exception(f"Category not found: {p}")
                color = possible_colors[index]
                colors.append(color)
            ax.scatter(y, x, s=0.1, c=colors)
            ax.set_xlabel(identifier_title)
            ax.set_ylabel("Delays (ms)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(min(50, len(set(y)))))
            ax.yaxis.set_major_locator(plt.MaxNLocator(min(30, len(set(x)))))
            plt.xticks(rotation=90)
            path = os.path.join(os.getcwd(), f"{file_title}.png")
            patches = [mpatches.Patch(color=col, label=cat) for col, cat in zip(possible_colors, cat_list)]
            plt.legend(handles=patches, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
            print("Saving data to: ", path)
            plt.tight_layout()
            plt.savefig(path)
            plt.clf()


def plot_spike_rate_data(path, file_title, identifier_title, identifiers, nd):
    fig, ax = plt.subplots()
    df = pd.read_csv(path)
    index = list(df.index[df["name"] == "count"])[0]
    df_rows= df.loc[:index - 1]
    names = list(df_rows["name"])
    data = [[] for x in range(df_rows.shape[0])]
    SR_keys = [k for k in list(df.keys()) if k.endswith("SR") and k.startswith("n")]
    for i, name in enumerate(names):
        f = []
        d = []
        for fx in identifiers:
            f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
        for dx in range(nd):
            d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
        sr = sum([x for x in [df_rows.iloc[i][srk] for srk in SR_keys]])
        config = (f + d)
        config.append(sr)
        [data[i].append(x) for x in config]
    data_sort = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                      range(len(identifiers))], 2)]),
                                                  np.mean([abs(x[0] - x[1]) for x in
                                                           itertools.combinations([element[x] for x in
                                                                                   range(len(identifiers), len(identifiers) + nd)], 2)]),
                                                  element[:len(identifiers) + nd]))

    x = []
    y = []
    z = []
    for row in data_sort:
        x_str = ""
        y_str = ""
        for yid in range(len(identifiers)):
            if yid < len(identifiers) - 1:
                y_str += f"{row[yid]}-"
            else:
                y_str += f"{row[yid]}"
        for xid in range(len(identifiers), nd + len(identifiers)):
            if xid < nd + len(identifiers) - 1:
                x_str += f"{row[xid]}-"
            else:
                x_str += f"{row[xid]}"
        x.append(x_str)
        y.append(y_str)
        z.append(row[-1])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(z))
    ax.scatter(y, x, s=0.1, c=z, cmap=plt.cm.get_cmap("Reds"))
    ax.set_ylabel("Delays (ms)")
    ax.set_xlabel(identifier_title)
    plt.xticks(rotation=90)
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(50, len(set(y)))))
    ax.yaxis.set_major_locator(plt.MaxNLocator(min(30, len(set(x)))))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("Reds")), ax=ax)
    plt.tight_layout()
    path = os.path.join(os.getcwd(), f"{file_title}.png")
    print("Saving data to: ", path)
    plt.savefig(path)


def reduce_sim_file_size(dir):
    for dirs, subdirs, files in os.walk(dir):
        if "synapse_data.json" in files:
            file = os.path.join(dirs, "synapse_data.json")
            new_data = {}
            with open(file, "r") as json_file:
                try:
                    data = json.load(json_file)
                except:
                    print("Unable to load:", file)
                if any([k for k in data.keys() if type(data[k]["d_hist"]) == dict]):
                    for k in data.keys():
                        if type(data[k]["d_hist"]) == dict:
                            data[k]["d_hist"] = data[k]["d_hist"]["d"]
                    new_data = data
            if new_data:
                os.remove(file)
                with open(file, "w") as new_file:
                    json.dump(data, new_file)
                    print("Made changes to: ", file)

def change_name(dir, string_to_remove):
    # MÅ FIKSES!!!
    main_dir = os.path.split(dir)[0]
    if re.search(string_to_remove,dir):
        sd = dir.replace(string_to_remove, "")
        dd = os.path.join(main_dir, sd)
        print("before: ", dir)
        try:
            os.rename(os.path.join(main_dir,dir),dd)
            print("after: ", dd)
        except:
            print(dir)


def plot_delay_catetgories_heatmap(dir, file_title, identifier_title, identifiers, nd):
    cat_list = C.DELAY_CATEGORIES_SHORTLIST
    possible_colors = C.DCAT_COLORS[:len(cat_list)]
    df = pd.read_csv(os.path.join(dir, "simulation_data.csv"))
    index = list(df.index[df["name"] == "count"])[0]
    df_rows = df.loc[:index - 1]
    data = [[] for x in range(df_rows.shape[0])]
    for col in df_rows.keys():
        if col == "name":
            names = list(df_rows[col])
            for i, name in enumerate(names):
                f = []
                d = []
                for fx in identifiers:
                    f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
                for dx in range(nd):
                    d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
                config = (f + d)
                [data[i].append(x) for x in config]
        elif re.match("[0-9]+_[0-9]+$", col):
            conn = list(df_rows[col])
            for i, c in enumerate(conn):
                data[i].append(c)
    data = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                               range(len(identifiers))], 2)]),
    np.mean([abs(x[0] - x[1]) for x in
             itertools.combinations([element[x] for x in
                                     range(len(identifiers), len(identifiers) + nd)], 2)]),
    element[:len(identifiers) + nd]))
    x = []
    y = []
    z = []
    dct = {}
    for row in data:
        x_str = ""
        y_str = ""
        cat = []
        for yid in range(len(identifiers)):
            if yid < len(identifiers) - 1:
                y_str += f"{row[yid]}-"
            else:
                y_str += f"{row[yid]}"
        for xid in range(len(identifiers), nd + len(identifiers)):
            if xid < nd + len(identifiers) - 1:
                x_str += f"{row[xid]}-"
            else:
                x_str += f"{row[xid]}"
        x.append(x_str)
        y.append(y_str)
        for catid in range(nd + len(identifiers), len(row)):
            cat.append(row[catid])
        if "dormant" in cat:
            z.append("dormant")
            if x_str not in dct.keys():
                dct[x_str] = {}
            dct[x_str][y_str] = cat_list.index("dormant")
        elif any(i in cat for i in ["increasing", "decreasing", "min", "max"]):
            z.append("diverging")
            if x_str not in dct.keys():
                dct[x_str] = {}
            dct[x_str][y_str] = cat_list.index("diverging")
        elif "uncategorized" in cat:
            z.append("uncategorized")
            if x_str not in dct.keys():
                dct[x_str] = {}
            dct[x_str][y_str] = cat_list.index("uncategorized")
        elif "repeating" in cat:
            z.append("repeating")
            if x_str not in dct.keys():
                dct[x_str] = {}
            dct[x_str][y_str] = cat_list.index("repeating")
        elif "converging" in cat:
            z.append("converging")
            if x_str not in dct.keys():
                dct[x_str] = {}
            dct[x_str][y_str] = cat_list.index("converging")
        else:
            raise Exception(f"Category {cat} not found!")
    dff = pd.DataFrame(dct)
    dff = dff.loc[::-1]
    fig, ax = plt.subplots()
    im = ax.imshow(dff, cmap=matplotlib.colors.ListedColormap(["g", "r", "b", "y", "black"]), interpolation="none", aspect='auto', vmin=0, vmax=4)
    ax.set_ylabel(identifier_title)
    ax.set_xlabel("Delays (ms)")
    ax.set_xticks(range(len(dff.keys())))
    ax.set_yticks(range(len(dff.index)))
    ax.set_xticklabels(dff.keys())
    ax.set_yticklabels(dff.index)
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(50, len(set(y)))))
    ax.yaxis.set_major_locator(plt.MaxNLocator(min(15, len(set(x)))))
    ax.grid(False)
    plt.xticks(rotation=90)
    colors = [im.cmap(im.norm(value)) for value in range(len(C.DELAY_CATEGORIES_SHORTLIST))]
    patches = [mpatches.Patch(color=col, label=cat.capitalize()) for col, cat in zip(colors, cat_list)]
    plt.legend(handles=patches, ncol=5, fontsize=9,loc="upper center", bbox_to_anchor=(0.4, 1.2), handletextpad=0.2)
    plt.tight_layout()
    plt.savefig(file_title, bbox_inches='tight')



def plot_SR_heatmap(path, file_title, identifier_title, identifiers, nd):
    df = pd.read_csv(os.path.join(path, "simulation_data.csv"))
    index = list(df.index[df["name"] == "count"])[0]
    df_rows = df.loc[:index - 1]
    names = list(df_rows["name"])
    data = [[] for x in range(df_rows.shape[0])]
    SR_keys = [k for k in list(df.keys()) if k.endswith("SR") and k.startswith("n")]
    for i, name in enumerate(names):
        f = []
        d = []
        for fx in identifiers:
            f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
        for dx in range(nd):
            d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
        sr = np.mean([x for x in [df_rows.iloc[i][srk] for srk in SR_keys]])
        config = (f + d)
        config.append(sr)
        [data[i].append(x) for x in config]
    data_sort = sorted(data, key=lambda element: (
    np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                               range(len(identifiers))], 2)]),
    np.mean([abs(x[0] - x[1]) for x in
             itertools.combinations([element[x] for x in
                                     range(len(identifiers), len(identifiers) + nd)], 2)]),
    element[:len(identifiers) + nd]))

    dct = {}
    for row in data_sort:
        x_str = ""
        y_str = ""
        for yid in range(len(identifiers)):
            if yid < len(identifiers) - 1:
                y_str += f"{row[yid]}-"
            else:
                y_str += f"{row[yid]}"
        for xid in range(len(identifiers), nd + len(identifiers)):
            if xid < nd + len(identifiers) - 1:
                x_str += f"{row[xid]}-"
            else:
                x_str += f"{row[xid]}"
        if not x_str in dct.keys():
            dct[x_str] = {}
        dct[x_str][y_str] = row[-1]

    dff = pd.DataFrame(dct)
    dff = dff.loc[::-1]
    fig, ax = plt.subplots()
    V_MAX = 60
    im = ax.imshow(dff, cmap=plt.cm.get_cmap("Reds"), interpolation="none", aspect='auto', vmin=0, vmax=V_MAX)
    ax.set_ylabel(identifier_title)
    ax.set_xlabel("Delays (ms)")
    ax.set_xticks(range(len(dff.keys())))
    ax.set_yticks(range(len(dff.index)))
    ax.set_xticklabels(dff.keys())
    ax.set_yticklabels(dff.index)
    ax.xaxis.set_major_locator(plt.MaxNLocator(50))
    ax.yaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(False)
    plt.xticks(rotation=90)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=V_MAX)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("Reds")), ax=ax)
    cb.set_label('spikes/s', rotation=90, labelpad=5)
    plt.tight_layout()
    plt.savefig(file_title, bbox_inches='tight')

def create_dataset(samples, input_length, seed):
    rng = np.random.default_rng(seed)
    x = []
    y = []
    for n in range(samples):
        type = np.random.choice(["alternating", "repeating", "asynchronous"])
        if type == "alternating":
            pass
        if type == "repeating":
            pass
        if type == "asynchronous":
            freq = rng.integers(1, 10)
            xi = [x for x in range(1, input_length) if x % freq == 0]
            x.append(xi)
            y.append()


def create_alternating_input(i, l, l_pattern=500, l_interm=500):
    pattern = []
    period1 = np.random.randint(30, 61)
    period2 = np.random.randint(30, 61)
    for x in range(i):
        offset1 = np.random.randint(0, 11)
        pattern1 = [(offset1 + (period1 * rep)) for rep in range(l_pattern) if (offset1 + (period1 * rep) < l_pattern)]
        offset2 = np.random.randint(0, 11)
        pattern2 = [(offset2 + (period2 * rep)) for rep in range(l_pattern) if (offset2 + (period2 * rep) < l_pattern)]
        p = []
        flip = 1
        for rnd in range(int(l / (l_pattern + l_interm)) + 1):
            [p.append(inp + (l_pattern + l_interm) * rnd) for inp in (pattern1 if flip == 1 else pattern2)]
            flip *= -1
        pattern.append(p)
    return pattern

def create_asynchronous_input(i, l):
    pattern = []
    for input in range(i):
        interval = np.random.randint(20,50)
        p = [x for x in range(1, l) if x % interval == 0]
        pattern.append(p)
    return pattern

def create_repeating_input(i , l, interval=None, seed=None):
    pattern = []
    if seed is not None:
        rng = np.random.default_rng(seed)
        if interval is None:
            interval = rng.integers(30, 61)
        for input in range(i):
            offset = rng.integers(0, 10)
            p = [offset + (interval * x) for x in range(1,int(np.ceil((l - offset) / interval))) if offset + (interval * x) < l]
            pattern.append(p)
        return pattern
    else:
        if interval is None:
            interval = np.random.randint(30, 61)
        for input in range(i):
            offset = np.random.randint(0, 10)
            p = [offset + (interval * x) for x in range(1,int(np.ceil((l - offset) / interval))) if offset + (interval * x) < l]
            pattern.append(p)
        return pattern


def get_polygroup_depth(polygroup, level=1):
    if not isinstance(polygroup, dict) or not polygroup:
        return level
    return max(get_polygroup_depth(polygroup[k], level + 1)
               for k in polygroup.keys())

def get_input_times(polygroup):
    for v in polygroup.values():
        if isinstance(v, dict):
            yield from get_input_times(v)
        else:
            yield v




def old_compare_poly(l1, l2):
    global match, old_unique
    match = 0
    old_unique = 0
    print(l1, l2)
    def compare(l1, l2):
        global match, old_unique
        if isinstance(l1, dict) and isinstance(l2, dict):
            intersect = set.intersection(set(l1.keys()), set(l2.keys()))
            old_unique += len(set(list(l1.keys()) + list(l2.keys())))
            match += len(set.intersection(set(l1.keys()), set(l2.keys())))
            for k in intersect:
                compare(l1[k], l2[k])
    inputs_intersect = set.intersection(set(l1.keys()), set(l2.keys()))
    for ii in inputs_intersect:
        compare(l1[ii], l2[ii])
    return match, old_unique

def print_polygroup(d, indent=0, s = ""):
   for key, value in d.items():
      s += '\n' + '-' * indent + str(key)
      if isinstance(value, dict):
         s += print_polygroup(value, indent+1, s=s)
      else:
         s += '\n' + '-' * (indent+1) + str(value)
   return s

def create_class_diagram(file=None, cls=None):
    if file is not None:
        os.system(f"pyreverse -o png -p MasterThesis {file}Population.py")
    elif file is not None and cls is not None:
        os.system(f"pyreverse -o png -p MasterThesis {file}Population.py -c {cls}")
    elif file is None and cls is not None:
        raise Exception("Must specify file.")
    else:
        os.system(f"pyreverse -o png -p MasterThesis .")


def create_mnist_input(sample_size, numbers, interval, image_size=10):
    raise Exception("Må fikses. Gir flere input per input node på samme tid.")
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    img = []
    for n in numbers:
        img.append([np.round(cv2.resize(x, (image_size, image_size)) * 4 / 25.5, 1) for x, y in zip(train_X, train_y) if y == n][:sample_size])
    i = img[0][0].size
    input = [[] for _ in range(i)]
    intv = 0
    for num in img:
        for samp in num:
            flt = samp.flatten()
            for inp in range(i):
                input[inp].append(flt[inp] + intv)
        intv += interval
    return input

def create_random_mnist_input(sample_size, numbers, interval, image_size=10):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    img = []
    for x, y in zip(train_X, train_y):
        if len(img) >= sample_size:
            break
        elif y in numbers:
            img.append([np.round(cv2.resize(x, (image_size, image_size)) * 4 / 25.5, 1)])
    i = img[0][0].size
    input = [[] for _ in range(i)]
    intv = 0
    for inst in img:
        for samp in inst:
            flt = samp.flatten()
            for inp in range(i):
                input[inp].append(flt[inp] + intv)
        intv += interval
    return input


def create_mnist_sequence_input(train_sequence, test_sequence, interval, breaks, image_size=10):
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    #TEST
    test_img = []
    test_seq = test_sequence.copy()
    for x, y in zip(train_X, train_y):
        if not test_seq:
            break
        elif test_seq[0] == y:
            img = np.array(np.round(cv2.resize(x, (image_size, image_size)) * 4 / 25.5, 1))
            flattened_img = img.flatten()
            test_img.append(flattened_img)
            test_seq.pop(0)

    #TRAIN
    train_img = []
    train_seq = train_sequence.copy()
    for x, y in zip(train_X, train_y):
        if not train_seq:
            break
        elif train_seq[0] == y:
            img = np.array(np.round(cv2.resize(x, (image_size, image_size)) * 4 / 25.5, 1))
            flattened_img = img.flatten()
            train_img.append(flattened_img)
            train_seq.pop(0)

    img_list = test_img + train_img + test_img
    input = [[] for _ in range(image_size**2)]
    intv = 0
    for j, inst in enumerate(img_list):
        for inp, pixl in zip(input, inst):
            inp.append(pixl + intv)
        if j + 1 in breaks:
            intv += 2*interval
        else:
            intv += interval
    return input




def create_mnist_input_from_file(sample_size, numbers, interval, image_size=10):
    from mlxtend.data import loadlocal_mnist
    import platform
    if not platform.system() == 'Windows':
        train_X, train_y = loadlocal_mnist('Data/train-images-idx3-ubyte', 'Data/train-labels-idx1-ubyte')
    else:
        train_X, train_y = loadlocal_mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')
    img = []
    for n in numbers:
        img.append([np.round(cv2.resize(np.reshape(x, (28, 28)), (image_size, image_size)) * 4 / 25.5, 1) for x, y in zip(train_X, train_y) if
                    y == n][:sample_size])
    i = img[0][0].size
    input = [[] for _ in range(i)]
    intv = 0
    for num in img:
        for samp in num:
            flt = samp.flatten()
            for inp in range(i):
                input[inp].append(flt[inp] + intv)
            intv += interval
        intv += interval
    return input

def save_image(matrix, out_dir):
    if not isinstance(matrix ,np.ndarray):
        matrix = np.asarray(matrix)
    im = Image.fromarray(matrix)
    n = 0
    pth = os.path.join(out_dir, "image.jpeg")
    while os.path.exists(pth):
        pth = os.path.join(out_dir, f"image{n}.jpeg")
        n += 1
    im = im.convert("L")
    im.save(pth)

def compile_results(dir, name, n_test_digits, n_test_inst, n_train_inst):
    workbook_name = f"{name}.xlsx"
    wb = xlsxwriter.Workbook(workbook_name)
    ws = wb.add_worksheet()
    title_f = wb.add_format()
    title_f.set_bold()
    title_f.set_align('center')
    title_f.set_align('vcenter')
    f = wb.add_format()
    f.set_align('right')
    f.set_align('vcenter')
    ws.write_row(0, 0, ["Image size", "Layers", "Train inst.", "Train digits", "Test inst.", "Test digits", "Weight", "Threshold", "P", "Seed", "Avg SR (spike/s)", "Memory (MB)", "Sim time (min)"], title_f)

    memory_mean = []
    duration_mean = []
    mean_acc = [[0] for _ in range(2*n_test_digits + 2)]
    pre_acc = []
    post_acc = []
    dirlist = [d for d in os.listdir(dir) if "TrainingPhase" in d]
    for row, sim in enumerate(dirlist):
        print(sim)
        pre_acc.clear()
        post_acc.clear()
        pre_mc = []
        post_mc = []
        img = re.findall("img-(\d+)", sim)[0]
        layers = re.findall("layers-(\d+)", sim)[0]
        train_inst = re.findall("train_inst-(\d+)", sim)[0]
        train_digits = re.findall("traindigits-(\[.+\])_test", sim)[0]
        test_inst = re.findall("test_inst-(\d+)", sim)[0]
        test_digits = re.findall("testdigits-(\[.+\])", sim)[0]
        w = re.findall("w-(\d+)", sim)[0]
        th = re.findall("th-(\d+\.\d+)", sim)[0]
        p = re.findall("p-(\d+\.\d+)", sim)[0]
        seed = re.findall("seed-(\d+)", sim)[0]
        if os.path.exists(os.path.join(dir, sim, "neuron_data.json")):
            neuron_data = json.load(open(os.path.join(dir, sim, "neuron_data.json")))
            avg_SR = 0
            for id in neuron_data:
                if neuron_data[id]["type"] == "<class 'Population.Population.RS'>":
                    spikes = len(neuron_data[id]["spikes"])
                    SR = spikes/(neuron_data[id]["duration"]/1000)
                    avg_SR += SR
            avg_SR = round(avg_SR/len(neuron_data),1)
        else:
            avg_SR = "n/a"
        if os.path.exists(os.path.join(dir, sim, "SimStats.txt")):
            simstats = open(os.path.join(dir, sim, "SimStats.txt")).read()
            memory = re.findall("usage: (.+)MB", simstats)[0]
            memory_mean.append(float(memory))
            duration = re.findall("time: (.+)min", simstats)[0]
            duration_mean.append(float(duration))
        else:
            memory = "n/a"
            duration = "n/a"
        if os.path.exists(os.path.join(dir, sim, "PG_data.json")):
            acc_stats = json.load(open(os.path.join(dir, sim, "PG_data.json")))
            pps = []

            for i,pg in acc_stats.items():
                for pp in pg['pps']:
                    pps.append((i, pp['pattern_start']))
            pps.sort(key=lambda x: x[1])

            pps = [x[0] for x in pps]

            pre_mc.clear()
            post_mc.clear()

            for c in range(0, n_test_inst*n_test_digits, n_test_inst):
                mc = Counter(pps[c:c + n_test_inst]).most_common(1)[0]

                pre_acc.append(np.round(mc[1] / n_test_inst, 2))
                pre_mc.append(mc[0])

            for c in range(n_test_inst*n_test_digits + n_train_inst, n_test_inst*n_test_digits*2 + n_train_inst, n_test_inst):
                mc = Counter(pps[c:c+n_test_inst]).most_common(1)[0]

                post_acc.append(np.round(mc[1]/n_test_inst,2))
                post_mc.append(mc[0])

        ws.write_row(row + 1, 0, [img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, seed, avg_SR, memory, duration], f)
        acc = np.round(np.mean(pre_acc),2) if len(pre_mc)==len(set(pre_mc)) else "n/s"
        mean_acc[0].append(acc if isinstance(acc, float) else 0)
        ws.write_row(row + 1, 13, [acc], f)
        col = 14
        for pre, mc in zip(pre_acc, pre_mc):
            acc = pre if pre_mc.count(mc) == 1 else "n/s"
            mean_acc[col - 14].append(acc if isinstance(acc, float) else 0)
            ws.write_row(row + 1, col, [acc], f)
            col += 1
        acc = np.round(np.mean(post_acc),2) if len(post_mc) == len(set(post_mc)) else "n/s"
        mean_acc[col - 14].append(acc if isinstance(acc, float) else 0)
        ws.write_row(row + 1, col, [acc], f)
        col += 1
        for post, mc in zip(post_acc, post_mc):
            acc = post if post_mc.count(mc) == 1 else "n/s"
            mean_acc[col - 14].append(acc if isinstance(acc, float) else 0)
            ws.write_row(row + 1, col, [acc], f)
            col += 1

        if os.path.exists(os.path.join(dir, sim, "spikes.png")):
            ws.insert_image(row + 1, col, os.path.join(dir, sim, "spikes.png"), {'x_scale': 0.7, 'y_scale': 0.5})
        else:
            ws.write_row(row + 1, col, "n/a")
        ws.set_column(col, col, 20)
        ws.set_row(row + 1, 200)
    ws.write_row(0, 13, ["Pre total"] + [f"Pre {x}" for x in range(len(pre_acc))] + ["Post total"] + [f"Post {y}" for y in range(len(post_acc))] + ["Raster plot"], title_f)
    ws.write_row(len(dirlist)+1, 0, ["Mean"])
    ws.write_row(len(dirlist)+1, 13, [np.round(np.mean(x),2) for x in mean_acc])
    ws.write_row(len(dirlist)+1, 11, [np.mean(memory_mean), np.mean(duration_mean)])
    wb.close()

def simplify_neuron_data(dir):
    for sim in os.listdir(dir):
        print("Simplifying neuron data: ", sim)
        if os.path.isdir(os.path.join(dir, sim)):
            with open(os.path.join(dir, sim, "neuron_data.json")) as f:
                data = json.load(f)
                simplified_data = {}
                for id in data.keys():
                    simplified_data[id] = data[id]["spikes"]
                output_file = open(os.path.join(dir, sim, "simplified_neuron_data.json"), 'w')
                json.dump(simplified_data, output_file)
                output_file.close()

def simplify_PG_data(dir):
    for sim in os.listdir(dir):
        print("Simplifying PG data: ", sim)
        if os.path.isdir(os.path.join(dir, sim)):
            with open(os.path.join(dir, sim, "PG_data.json")) as f:
                data = json.load(f)
                simplified_data = []
                for pg_id in data.keys():
                    for pp in data[pg_id]["pps"]:
                        simplified_data.append((pp["pattern_start"], pg_id))
                sorted_simplified_data = sorted(simplified_data, key=lambda x : x[0])
                sorted_ids = [x[1] for x in sorted_simplified_data]
                output_file = open(os.path.join(dir, sim, "simplified_pg_data.json"), 'w')
                json.dump(sorted_ids, output_file)
                output_file.close()

def change_threshold(ddir, dir, th_from, th_to):
    if os.path.isdir(os.path.join(ddir, dir.replace(f'th-{th_from}', f'th-{th_to}'))):
        return
    m = load_model(os.path.join(os.path.join(ddir,dir),"post_sim_model.pkl"))
    m.dir = os.path.join(ddir, dir)
    m.build_pgs(min_threshold=th_to)
    m.save_PG_data()
    m.plot_raster()
    save_model(m, os.path.join(ddir,os.path.join(dir, "post_sim_model.pkl")))
    os.rename(os.path.join(ddir,dir), os.path.join(ddir, dir.replace(f'th-{th_from}', f'th-{th_to}')))

def copy_rename_neuron_data(dir):
    new_dir = os.path.join(dir, f"neuron_data_{os.path.split(dir)[1]}")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for sim in os.listdir(dir):
        print("Copy/rename neuron data: ", sim)
        if sim == os.path.split(new_dir)[1] or os.path.splitext(sim)[1] == ".xlsx":
            continue
        print(sim)
        seed = re.findall('seed-(\d+)', sim)[0]
        dest = os.path.join(new_dir, f"neuron_data_{seed}.json")
        src = os.path.join(dir, sim, "simplified_neuron_data.json")
        if not os.path.isfile(dest):
            shutil.copy(src, dest)

def copy_rename_pg_data(dir):
    new_dir = os.path.join(dir, f"pg_data_{os.path.split(dir)[1]}")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for sim in os.listdir(dir):
        print("Copy/rename PG data: ", sim)
        if not sim.startswith("TrainingPhase"):
            continue
        print(sim)
        seed = re.findall('seed-(\d+)', sim)[0]
        dest = os.path.join(new_dir, f"pg_data_{seed}.json")
        src = os.path.join(dir, sim, "simplified_pg_data.json")
        if not os.path.isfile(dest):
            shutil.copy(src, dest)