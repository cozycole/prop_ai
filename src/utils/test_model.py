import torch
from torch import nn
from src import my_model as mm
from src import load_data as ld
from config import class_list as classes
import matplotlib.pyplot as plt

model = mm.NewModel("/home/colet/programming/projects/ai_model/src/resnet18_places365.pth.tar")
model.load_state_dict(torch.load("/home/colet/programming/projects/ai_model/model_data/new_model"))
model.to("cuda")
loss_fn = nn.CrossEntropyLoss()

loader = ld.test_loader

# Here we want to test the threshold needed to include
# all distressed houses (to see how far you'd need to 
# diverge from 0.333)
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
size = len(loader.dataset)
num_batches = len(loader)
test_loss, correct = 0, 0

distress_distrib = {
    ".33+" : 0,
    ".33-.3" : 0,
    ".3-.2" : 0,
    ".2-" : 0
}

model.eval()
with torch.no_grad():
    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        predictions = model(X)
        test_loss += loss_fn(predictions, y).item()
        predictions = nn.functional.softmax(predictions, dim=1)
        y = y.argmax(1)
        for label, pred in zip(y, predictions):
            if label == 0 :
                if pred[0] >= .33:
                    distress_distrib[".33+"] += 1
                elif .33 > pred[0] >= .3:
                    distress_distrib[".33-.3"] += 1
                elif .3 > pred[0] >= .2:
                    distress_distrib[".3-.2"] += 1
                else:
                    distress_distrib[".2-"] += 1
            pred = pred.argmax(0)
            if label == pred:
                correct_pred[classes[label]] += 1
                correct += 1
            total_pred[classes[label]] += 1
    test_loss /= num_batches
    correct /= size
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} % (Total: {total_pred[classname]})')
    print(f"Total Test Set Stats: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print(distress_distrib)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
# ax.bar(distress_distrib.keys(), distress_distrib.values())
# plt.show()