import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, backends
from src import load_data as ld, train as t, my_model as mm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision

##### ENV VARS #####
dset_root = "/home/colet/programming/projects/ai_model/data"
path = "src/resnet18_places365.pth.tar"
labels_map = {
    0: "distress",
    1: "no_distress",
    2: "unknown"
}

classes = ["distress", "no_distress", "unknown"]
seed = 0
####################

##### HYPERPARAMS #####
learning_rate = 0.0003
batch_size = 64
epochs = 100
#######################

model = mm.NewModel(path)
load_model = ""
loaded_path = os.path.join(os.getcwd(), "model_data", load_model)
if load_model and os.path.exists(loaded_path):
    model.load_state_dict(torch.load(loaded_path))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
torch.manual_seed(seed)

writer = SummaryWriter("runs/test1")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

img_trans = transforms.Compose([
        transforms.CenterCrop((800,1000)),
        transforms.Resize((600, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

train_dataset = datasets.ImageFolder(
    root=os.path.join(dset_root, "train"),
    transform=img_trans,
    target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

valid_dataset = datasets.ImageFolder(
    root=os.path.join(dset_root, "valid"),
    transform=img_trans,
    target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

test_dataset = datasets.ImageFolder(
    root=os.path.join(dset_root, "test"),
    transform=img_trans,
    target_transform=transforms.Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))


train_loader = DataLoader(train_dataset ,batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset ,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset ,batch_size=batch_size, shuffle=True)

### Testing with one batch
# data, target = next(iter(train_loader))
# img_grid = torchvision.utils.make_grid(data)
# writer.add_image("first_batch_images", img_grid)
# writer.add_graph(model, data.cuda())
# writer.close()

print("##### Training Set Image Distribution #####")
data_distrib = ld.get_img_distrib(os.path.join(dset_root, "train"), classes)
total = sum(list(data_distrib.values()))
for key, val in data_distrib.items():
    print(f"Class {key}: {round(100 * val/total,2)}% ({val} images)")

print("##### Test Set Image Distribution #####")
data_distrib = ld.get_img_distrib(os.path.join(dset_root, "test"), classes)
total = sum(list(data_distrib.values()))
for key, val in data_distrib.items():
    print(f"Class {key}: {round(100 * val/total,2)}% ({val} images)")

print("test new model")
t.test_loop(test_loader, model, loss_fn, classes)
input("press enter to continue")

# plotting loss graphs
valid_points = []
train_points = []
epoch_points = []

try:
    size = len(train_loader.dataset)
    last_loss = 100
    patience = 3
    stop_count = 0
    early_stop = t.EarlyStopping()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch, (data, target) in loop:
            X, y = data.cuda(), target.cuda()
            
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (pred.argmax(1) == y.argmax(1)).sum().item()
            accuracy = correct / batch_size
            # print("Accuracy: ",accuracy)
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss = loss, accuracy=accuracy)
        
        
        # writer.add_scalar("Loss/train(extra layer)", train_loss / (batch + 1), epoch)
        train_loss /= (batch + 1)
        train_points.append(train_loss)
        epoch_points.append(epoch+1)
        valid_loss = t.valid_loop(valid_loader, model, loss_fn, classes)
        valid_points.append(valid_loss)

        early_stop(train_loss, valid_loss)
        if early_stop.early_stop:
            print("Ending training at epoch:", epoch)
            break

except KeyboardInterrupt:
    print("interrupt detected")

finally:
    t.test_loop(test_loader, model, loss_fn, classes)

    # ensure all list lengths are the same (for keyboard escape)
    min_len = min(len(epoch_points), len(train_points), len(valid_points))
    epoch_points, train_points, valid_points = epoch_points[:min_len], train_points[:min_len], valid_points[:min_len]
    
    # create loss graph
    plt.plot(epoch_points, train_points, 'g', label='Training loss')
    plt.plot(epoch_points, valid_points, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_graph.png")

    torch.save(model.state_dict(), "model_data/new_model")

    writer.flush()
    writer.close()

# data, targets = next(iter(ld.train_loader))
# for i in range(epochs):
#   print(f"Epoch {i+1}\n--------------------------")
#   t.train_loop(ld.train_loader, model, loss_fn, optimizer,i, epochs)
#   t.test_loop(ld.test_loader, model, loss_fn, ["distress", "no_distress", "slight_distress", "unknown"])
# print("Done!")