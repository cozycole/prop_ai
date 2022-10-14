import torch
import torch.utils.data
import torchmetrics as tm
from tqdm import tqdm


def train_loop(dataloader, model: torch.nn.Module, loss_fn, optimizer, curr_epoch, epoch_total):
    model.train()
    size = len(dataloader.dataset)
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for batch, (X, y, paths) in loop:
        X, y = X.cuda(), y.cuda()
        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{curr_epoch}/{epoch_total}]")
        loop.set_postfix(loss = loss)
    print("Done Training")

def valid_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    model.eval()
    with torch.no_grad():
        for X, y, paths in dataloader:
            X, y = X.cuda(), y.cuda()
            predictions = model(X)
            test_loss += loss_fn(predictions, y).item()
            predictions = predictions.argmax(1)
            y = y.argmax(1)
            for label, pred in zip(y, predictions):
                if label == pred:
                    correct_pred[classes[label]] += 1
                    correct += 1
                total_pred[classes[label]] += 1
            # print(f"Predictions: ", predictions)
            # print(f"Labels: ", y)
            # print(recall(predictions, y).item())
            # input("continue?")
        test_loss /= num_batches
        correct /= size
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} % (Total: {total_pred[classname]})')
        print(f"Total Validation Stats: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def test_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    distress_recall_tp = 0
    distress_recall_fn = 0
    model.eval()
    with torch.no_grad():
        for X, y, paths in dataloader:
            X, y = X.cuda(), y.cuda()
            predictions = model(X)
            test_loss += loss_fn(predictions, y).item()
            predictions = predictions.argmax(1)
            y = y.argmax(1)
            for label, pred in zip(y, predictions):
                if label == pred:
                    correct_pred[classes[label]] += 1
                    correct += 1
                    if label == 0:
                        distress_recall_tp += 1
                else:
                    if label == 0:
                        distress_recall_fn += 1
                total_pred[classes[label]] += 1
            # print(f"Predictions: ", predictions)
            # print(f"Labels: ", y)
            # print(recall(predictions, y).item())
            # input("continue?")
        test_loss /= num_batches
        correct /= size
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} % (Total: {total_pred[classname]})')
        print(f"Total Test Set Stats: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class EarlyStopping():

    def __init__(self, tolerance=5, min_delta=0.05):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True