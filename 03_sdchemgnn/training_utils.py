import os
import os.path as osp
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import torch

from pathlib import Path
from chemgnn_backbone import MyCEALNetwork
from pymatgen.core import Structure


import re
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp
from pymatgen.io.cif import CifBlock




def worker_init_fn(worker_id):
    """Seed NumPy and PyTorch differently per DataLoader worker for reproducibility."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Folder created: {folder_path}')
    else:
        print(f'Folder already exists: {folder_path}')


def plot_training_progress(epochs_list, train_losses, val_losses, test_losses, title):
    """Live plot of training progress during the training loop."""
    lw, ms = 0.7, 0.7
    threshold = 1
    for i in range(len(train_losses)):
        train_losses[i] = min(train_losses[i], threshold)
        val_losses[i]   = min(val_losses[i],   threshold)
        test_losses[i]  = min(test_losses[i],  threshold)

    plt.plot(epochs_list, train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
    plt.plot(epochs_list, val_losses,   marker='s', linestyle='-', color='r', lw=lw, ms=ms)
    plt.plot(epochs_list, test_losses,  marker='*', linestyle='-', color='g', lw=lw, ms=ms)
    legend_entries = [
        Line2D([0], [0], color='blue',  label='train_loss'),
        Line2D([0], [0], color='red',   label='val_loss'),
        Line2D([0], [0], color='green', label='test_loss'),
    ]
    plt.title(title)
    plt.legend(handles=legend_entries, loc='upper right')
    plt.pause(0.001)


def save_mae(folder, maeFile, test_loss, msg):
    """Append (or create) a MAE result entry to maeFile."""
    result_file = osp.join(folder, maeFile)
    mode = 'a' if osp.exists(result_file) else 'w'
    with open(result_file, mode) as f:
        f.write(msg + '\n')
        f.write(str(test_loss) + '\n')


def save_train_progress(folder, filename, epochs, train_losses, val_losses, test_losses):
    """Save epoch loss history to a text file (one row per loss type)."""
    result_file = osp.join(folder, filename)
    with open(result_file, 'w') as f:
        f.write(' '.join(map(str, epochs))       + '\n')
        f.write(' '.join(map(str, train_losses)) + '\n')
        f.write(' '.join(map(str, val_losses))   + '\n')
        f.write(' '.join(map(str, test_losses))  + '\n')


def save_agg_weights(folder, filename, model):
    """Save softmax-normalised aggregator scalar weights to a text file."""
    result_file = osp.join(folder, filename)
    with open(result_file, 'w') as f:
        for name, param in model.named_parameters():
            if 'agg_weights' in name or 'aggregator_weights' in name:
                x = torch.nn.functional.softmax(param.data.clone(), dim=-1)
                x = x.cpu().numpy()
                f.write(f'Parameter name: {name}, {x}\n')


def save_regression_result(folder, filename, test_out, test_y):
    """
    Save predicted and true values side-by-side to a text file.
    test_out: predicted values (Tensor) — any shape reshapeable to (N, 1)
    test_y:   true values (Tensor)     — any shape reshapeable to (N, 1)

    FIX (shape robustness): the previous version did torch.unsqueeze(test_y, dim=1)
    unconditionally, which assumed a 1-D test_y. A caller passing test_y already
    shaped (N, 1) (e.g. a trainer's true_t.reshape(-1, 1)) got a
    (N, 1, 1) tensor and torch.cat then failed with
    "Tensors must have same number of dimensions: got 2 and 3".
    reshape(-1, 1) on BOTH tensors accepts (N,) and (N, 1) alike and is exactly
    equivalent to the old behaviour for the legacy 1-D convention.
    """
    result_file = osp.join(folder, filename)
    test_out = test_out.detach().reshape(-1, 1)
    test_y   = test_y.detach().reshape(-1, 1)
    if test_out.shape[0] != test_y.shape[0]:
        raise ValueError(
            f'save_regression_result: pred has {test_out.shape[0]} rows but '
            f'truth has {test_y.shape[0]}.')
    result  = torch.cat((test_out, test_y), dim=1).cpu().numpy()
    with open(result_file, 'w') as f:
        for row in result:
            f.write(' '.join(str(v) for v in row) + '\n')


def save_hyper_parameter(folder, hyper_para_filename, hyper_dict):
    """Save hyperparameter dictionary to a JSON file."""
    hyper_para_filename = osp.join(folder, hyper_para_filename)
    with open(hyper_para_filename, 'w') as f:
        json.dump(hyper_dict, f)


def load_regression_results(folder, filename):
    """
    Load regression results from a text file.
    Returns y (predicted) and x (true) as numpy arrays.
    """
    filepath = osp.join(folder, filename)
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = np.asarray([line.split() for line in data], dtype=float)
    y = data[:, 0]  # predicted
    x = data[:, 1]  # true
    return y, x


def get_density(x, y):
    """Kernel density estimate for each (x, y) point."""
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    return kernel(values)


def calculate_r_squared(y_predicted, y_actual):
    """Compute R^2 between predicted and actual numpy arrays."""
    ssr = np.sum((y_actual - y_predicted) ** 2)
    sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ssr / sst)


# ══════════════════════════════════════════════════════════════════════════════
# Local helpers — fixed versions of training_utils.py functions that depend on the
# old model interface (deg, newCeal) or on old filename patterns.
# ══════════════════════════════════════════════════════════════════════════════

def save_model_CEAL(model, folder, model_filename, model_para_filename):
    """
    Save model state_dict and constructor parameters to disk.
    Fixed from training_utils.py: removed references to model.deg and model.newCeal,
    which no longer exist in the cleaned MyCEALNetwork.
    """
    torch.save(model.state_dict(), osp.join(folder, model_filename))

    data = {
        'in_channels'  : model.in_channels,
        'out_channels' : model.out_channels,
        'aggregators'  : model.aggregators,
        'scalers'      : model.scalers,
        'numLayers'    : model.numLayers,
        'edge_dim'     : model.edge_dim,
        'towers'       : model.towers,
        'pre_layers'   : model.pre_layers,
        'post_layers'  : model.post_layers,
        'divide_input' : model.divide_input,
        'aggMLP'        : model.aggMLP,
        'aggMLP_factor' : model.aggMLP_factor,
        'dropout'       : model.dropout,
        'norm'          : model.norm,
    }
    with open(osp.join(folder, model_para_filename), 'w') as f:
        json.dump(data, f)


def plot_training_from_file(folder, filename, plotfilename=None, disp=True):
    """
    Plot training progress from a saved progress file.
    Fixed from training_utils.py: the original used a regex to extract dataset index
    strings (e.g. '12_34') from the filename for the plot title, which fails
    for our new prefix format. The title is now taken directly from the prefix
    embedded at the start of the filename.
    """
    # Use everything before the timestamp as the title
    title = filename.split('_training_progress')[0]

    result_file = osp.join(folder, filename)
    matrix = []
    with open(result_file, 'r') as f:
        for line in f:
            matrix.append(list(map(float, line.split())))

    epochs_list  = list(map(int, matrix[0]))
    train_losses = matrix[1]
    val_losses   = matrix[2]
    test_losses  = matrix[3]
    maxepoch     = max(epochs_list)

    # Clip losses to threshold for readability
    threshold = 3
    for i in range(len(epochs_list)):
        train_losses[i] = min(train_losses[i], threshold)
        val_losses[i]   = min(val_losses[i],   threshold)
        test_losses[i]  = min(test_losses[i],  threshold)

    lw, ms = 0.7, 0.7
    plt.figure(figsize=(8, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.plot(epochs_list, train_losses, marker='o', linestyle='-', color='b', lw=lw, ms=ms)
    plt.plot(epochs_list, val_losses,   marker='s', linestyle='-', color='r', lw=lw, ms=ms)
    plt.plot(epochs_list, test_losses,  marker='*', linestyle='-', color='g', lw=lw, ms=ms)
    legend_entries = [
        Line2D([0], [0], color='blue',  label='train_loss'),
        Line2D([0], [0], color='red',   label='val_loss'),
        Line2D([0], [0], color='green', label='test_loss'),
    ]
    plt.legend(handles=legend_entries, loc='upper right')

    if disp:
        plt.show()
    if plotfilename is not None:
        plt.savefig(osp.join(folder, plotfilename))
    plt.close()


def plot_regression_result(folder, filename, title, plotfilename=None, disp=False):
    """
    Scatter plot of predicted vs. true values with MAE and R^2 annotations.
    Pulled in from training_utils.py unchanged except removal of unused 'normalized'
    parameter and the two undefined helper calls it depended on.
    """
    y, x = load_regression_results(folder, filename)

    x_min, x_max = np.min(x) - 0.001, np.max(x) + 0.001
    y_min, y_max = np.min(y) - 0.001, np.max(y) + 0.001

    mae = np.mean(np.abs(x - y))
    ssr = np.sum((x - y) ** 2)
    sst = np.sum((x - np.mean(x)) ** 2)
    r2  = 1 - (ssr / sst)

    density = get_density(x, y)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=density, cmap='bwr', s=8, marker='o')
    plt.plot([min(x), max(x)], [min(x), max(x)], color='grey', linestyle='--')

    text_x = x_min + (x_max - x_min) * 0.7
    plt.text(text_x, y_min + (y_max - y_min) * 0.10, f'MAE = {mae:.6f}', fontsize=10)
    plt.text(text_x, y_min + (y_max - y_min) * 0.05, f'R^2 = {r2:.6f}',  fontsize=10)

    cbar = plt.colorbar(sc)
    cbar.set_label('Density')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)

    if plotfilename is not None:
        plt.savefig(osp.join(folder, plotfilename))
    if disp:
        plt.show()
    plt.close()


def load_ceal_model(checkpoint_file_name, device=None):
    # To load a MyCEALNetwork model from saved check point
    # Note: the check point file and the model constructor parameter file must follow this pattern:
    # checkpiont.pt checkpoint_para.json    
    # Usage:

    """
    model = load_ceal_model(
    "./Results/ChemGNN_06_15_2026_1/best_model.pt"
    )

    data = data.to(next(model.parameters()).device)

    with torch.inference_mode():
        pred = model(data)
    """


    checkpoint_path = Path(checkpoint_file_name)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model_para_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_para.json"
    )

    if not model_para_path.exists():
        raise FileNotFoundError(
            f"Model parameter file not found: {model_para_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    with open(model_para_path, "r") as f:
        para = json.load(f)

    model = MyCEALNetwork(
        in_channels=para["in_channels"],
        out_channels=para["out_channels"],
        aggregators=para["aggregators"],
        scalers=para["scalers"],
        numLayers=para["numLayers"],
        edge_dim=para["edge_dim"],
        towers=para["towers"],
        pre_layers=para["pre_layers"],
        post_layers=para["post_layers"],
        divide_input=para["divide_input"],
        aggMLP=para["aggMLP"],
        aggMLP_factor=para["aggMLP_factor"],
        dropout=para["dropout"],
        norm=para.get("norm", "layer"),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


def replace_symmetry_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    # FIX: pymatgen renamed as_xyz_string() -> as_xyz_str() (old name removed in current versions)
    ops = [op.as_xyz_str() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    # ROBUSTNESS: tolerate leading whitespace / extra spaces around the identity line
    # (matches both canonical "1 'x, y, z'" and indented "  1  'x, y, z'")
    # FIX: CrystaLLM indents the loop-header tags with a leading space
    # (" _symmetry_equiv_pos_site_id"); the original pattern required the tags
    # flush-left, so it never matched and the operators were left un-expanded
    # (every declared non-P1 cell silently collapsed to the identity op).
    # [ \t]* after each newline tolerates that indentation.
    pattern = r"loop_\n[ \t]*_symmetry_equiv_pos_site_id\n[ \t]*_symmetry_equiv_pos_as_xyz\n\s*1\s+'x, y, z'"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated


def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")


def remove_atom_props_block(cif):
    pattern = re.compile(r"(data_[^\n]*\n)loop_[\s\S]*?(_symmetry_space_group_name_H-M)", re.MULTILINE)
    new_cif = re.sub(pattern, r"\1\2", cif)
    return new_cif


def postprocess(cif: str, fname: str) -> str:
    try:
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}")

    return cif


def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True

