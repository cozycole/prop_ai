import torch
from torch import nn
from torchvision import datasets, models, transforms

labels_map = {
    0: "distress",
    1: "slight_distress",
    2: "no_distress",
    3: "unknown"
}

class NewModel(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super(NewModel, self).__init__()
        # import the trained model
        model = models.resnet18(num_classes=365)
        checkpoint = torch.load(pretrain_model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        
        # change prediction class count
        model.fc = nn.Linear(model.fc.in_features, 3)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = NewModel("/home/colet/programming/projects/ai_model/src/resnet18_places365.pth.tar")
    for i, layer in enumerate(model.model.children()):
        print(f"Layer {i}: {layer}")
        input("enter to next layer")