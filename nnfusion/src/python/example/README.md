# How to use NNFusion Python interface for inference/training

NNFusion Python interface wraps NNFusion CLI, targeting on seamless experience for PyTorch users.
With the interface, it's easy to feed a PyTorch model to NNFusion, then leverage NNFusion optimization to accelerate inference or training.

## Add NNFusion CLI and Python interface

Currently NNFusion doesn't support install from PIP, you need explicitly add it to your module search path.
If NNFusion CLI isn't installed, you should also explicitly add it.

```python
os.environ["PATH"] = os.path.abspath("/path/to/nnfusion_cli") + ":" + os.environ["PATH"]
sys.path.insert(1, os.path.abspath("{nnfusion_root}/src/python"))
```

## Inference

Here we assume a PyTorch model like mnist MLP is already defined. Then a PyTorch inference looks like

```python
model = MLP()
data = torch.ones([5, 1, 28, 28], dtype=torch.float32, device="cuda:0")
out = model(data)
```

Code below replaces the execution backend with NNFusion

```python
from nnfusion.runner import Runner

model = MLP()
data = torch.ones([5, 1, 28, 28], dtype=torch.float32, device="cuda:0")
runner = Runner(model)
out = runner(data)
```

You could compare NNFusion result against PyTorch.

## Training

There is no much difference for training, just replace Runner with Trainer

```python
from nnfusion.trainer import Trainer

model = MLP()
loss_func = F.nll_loss
device = "cuda:0"

trainer = Trainer(model, loss_func, device)
for i, batch in enumerate(train_loader):
    feed_data = [t.to(device) for t in batch]
    nnf_loss = trainer(*feed_data)

# save trained weights
torch.save(model.state_dict(), "/tmp/mnist.pt")
```

Then trained weights could be loaded, do inference in PyTorch or NNFusion

```python
model = MLP()
device = "cuda:0"
model.load_state_dict(torch.load("/tmp/mnist.pt"))
model.to(device)
model.eval()

data = torch.ones([5, 1, 28, 28], dtype=torch.float32, device="cuda:0")

# do inference in PyTorch
out = model(data)

# do inference in NNFusion
runner = Runner(model)
out = runner(data)
```

## Test

Run the mnist example directly to verfiy Python interface.

```bash
# make sure torch version
pip install torch==1.6.0 torchvision==0.7.0

# change to NNFusion root folder
cd /path/to/NNFusion

# run mnist example
python src/python/mnist.py
```
