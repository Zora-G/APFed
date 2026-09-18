import torch
from torch import nn
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, device=None):
    if device is None:
        if getattr(args, 'gpu', -1) >= 0 and torch.cuda.is_available():
            device = torch.device('cuda:' + str(args.gpu))
        else:
            device = torch.device('cpu')
    net_g.to(device)
    net_g.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loader = DataLoader(datatest, batch_size=128, shuffle=False)
    correct, loss, total = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net_g(images)
            loss += float(criterion(outputs, labels).item())
            prediction = torch.argmax(outputs, dim=1)
            correct += int(torch.sum(prediction == labels).item())
            total += int(labels.size(0))
    return 100.0 * correct / max(total, 1), loss / max(total, 1)
