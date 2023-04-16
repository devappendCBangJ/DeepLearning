import torch
from torchmetrics import MeanSquaredError

# Define training loop
def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanSquaredError()
    metric_mean = MeanSquaredError()

    for inputs, targets in loader:
        # 확인용 코드
        # print(type(inputs))
        # print(inputs.shape)
        # print(inputs)
        inputs = inputs.to(device)
        targets = targets.to(device)

        z, x_rec = model(inputs)
        loss = loss_fn(x_rec, targets)
        print(x_rec.shape)
        print(targets.shape)
        metric = metric_fn(x_rec, targets)
        print(x_rec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary

# Define evaluation loop
def evaluate(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanSquaredError()
    metric_mean = MeanSquaredError()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            z, x_rec = model(inputs)
        loss = loss_fn(x_rec, targets)
        metric = metric_fn(x_rec, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary