"""GPU check for SD-ChemGNN: the real CEAL backbone (no stand-in), the shipped
weights, synthetic crystal graphs. No dataset needed; needs a CUDA GPU.

    python test_cuda.py

Checks that the weights load, that GPU inference matches CPU, that training
steps run on the GPU, and that torch-scatter was built with CUDA support
(a CPU-only build fails here with "Not compiled with CUDA support")."""
import sys, os.path as osp, time
import torch
from torch_geometric.data import Data, Batch

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO, '03_sdchemgnn'))
from sdchemgnn_config import config_magnetic as cfg
from sdchemgnn_model import make_model_bs, SEED_CHANNELS

BASE_IN, EDGE_DIM = 95, 50
if not torch.cuda.is_available():
    sys.exit('No CUDA GPU visible to torch: torch.cuda.is_available() is False.')
dev = torch.device('cuda')
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'gpu', torch.cuda.get_device_name(0))


def graph(n, g):
    x = torch.rand(n, BASE_IN, generator=g)
    seed = torch.zeros(n, SEED_CHANNELS)
    seed[:, 0] = torch.randint(-1, 2, (n,), generator=g).float()   # sign
    seed[:, 1] = seed[:, 0].abs() * 2.0                              # formal |m|
    seed[:, 2] = 1.0                                                 # flag
    src = torch.arange(n).repeat_interleave(n - 1)
    dst = torch.cat([torch.cat([torch.arange(i), torch.arange(i + 1, n)])
                     for i in range(n)])
    ei = torch.stack([src, dst])
    return Data(x=torch.cat([x, seed], 1), edge_index=ei,
                edge_attr=torch.rand(ei.size(1), EDGE_DIM, generator=g),
                z_fu=torch.tensor([float(max(1, n // 4))]))


g = torch.Generator().manual_seed(0)
batch = Batch.from_data_list([graph(n, g) for n in (4, 8, 12, 16, 24, 6, 10, 20)])

model = make_model_bs(BASE_IN + SEED_CHANNELS, EDGE_DIM, cfg)
state = torch.load(osp.join(REPO, '03_sdchemgnn/trained_model/best_model.pt'),
                   map_location='cpu')
if isinstance(state, dict) and 'model' in state:
    state = state['model']
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, missing
n_par = sum(p.numel() for p in model.parameters())
print(f'[1] weights loaded: {n_par:,} params, unexpected={len(unexpected)}')

# [2] inference parity CPU vs GPU
model.eval()
batch_gpu = batch.clone().to(dev)
with torch.inference_mode():
    y_cpu = model.forward_b(batch)['y']
model.to(dev)
with torch.inference_mode():
    out = model.forward_b(batch_gpu)
    y_gpu = out['y'].cpu()
diff = (y_cpu - y_gpu).abs().max().item()
print(f'[2] GPU inference OK, y={y_gpu.numpy().round(3)}  max|cpu-gpu|={diff:.2e}')
assert torch.isfinite(y_gpu).all() and diff < 1e-3

# [3] training steps on GPU (forward + backward + optimizer)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
target = torch.randn(batch.num_graphs, device=dev)
for step in range(20):
    opt.zero_grad()
    o = model.forward_b(batch_gpu)
    loss = torch.nn.functional.l1_loss(o['y'], target)
    loss.backward()
    opt.step()
    if step in (0, 19):
        print(f'[3] train step {step:2d} loss {loss.item():.4f}')
grads_ok = all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())
assert grads_ok

# [4] throughput on a larger batch
big = Batch.from_data_list([graph(24, g) for _ in range(256)]).to(dev)
torch.cuda.synchronize(); t = time.time()
for _ in range(10):
    opt.zero_grad(); model.forward_b(big)['y'].sum().backward(); opt.step()
torch.cuda.synchronize()
print(f'[4] 256 graphs x 24 atoms: {(time.time() - t) / 10 * 1000:.1f} ms/step, '
      f'peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
print('ALL GPU CHECKS PASSED')
