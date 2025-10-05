from django.shortcuts import render
from .forms import ObjectForm
import lightkurve as lk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def min_max_normalize(tensor, min_range=-1, max_range=1):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
    return normalized_tensor


class ExoFinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(3197, 2000, bias=True)
        self.Matrix2 = nn.Linear(2000, 1000, bias=True)
        self.Matrix3 = nn.Linear(1000, 600, bias=True)
        self.Matrix4 = nn.Linear(600, 200, bias=True)
        self.Matrix5 = nn.Linear(200, 10, bias=True)
        self.Matrix6 = nn.Linear(10, 1, bias=True)
        self.R = nn.ReLU()
        self.S = F.tanh
    def forward(self, x):
        x = x.view(-1, 3197)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.R(self.Matrix3(x))
        x = self.R(self.Matrix4(x))
        x = self.R(self.Matrix5(x))
        x = self.R(self.Matrix6(x))
        x = self.S(x)
        return x.squeeze()


def base_view(request):
    error = None
    result = None
    if request.method == "POST":
        form = ObjectForm(request.POST)
        if form.is_valid():
            lc = lk.search_lightcurve(request.POST["object_id"], quarter=1).download()
            if lc is None:
                error = "Did not find an object"
            else:
                array = np.array(lc.flux, dtype=np.float32)
                state_dict = torch.load("./my_model.pth", weights_only=True)
                model = ExoFinder()
                model.load_state_dict(state_dict)
                model.eval()
                tensor = None
                if len(array) >= 3197:
                    tensor = torch.from_numpy(array[:3197])
                else:
                    tensor = torch.zeros(3197)
                    tensor[:len(array)] = torch.from_numpy(array)
                tensor = min_max_normalize(torch.nan_to_num(tensor, nan=0.0))
                print(tensor)
                print(model.Matrix1.weight)
                result = model(tensor).item()
                if result >= 0.05:
                    result = "It is an exoplanet"
                else:
                    result = "It is not an exoplanet (probably)"
        else:
            form = ObjectForm()
    else:
        form = ObjectForm()
    context = {
        "form": form,
        "result": result,
        "error": error,
    }
    return render(request, "kit.html", context)
