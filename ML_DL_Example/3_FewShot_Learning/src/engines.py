import torch

from torchmetrics.aggregation import MeanMetric


# 모델 예측값과 실제 결과값 비교(reshape 작업 안함) -> 학습(optimizer step, scheduler step) -> 결과 출력(loss, accuracy)
def pretrain_baseline(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


# 모델 예측값과 실제 결과값 비교 -> 학습(optimizer step, scheduler step) -> 결과 출력(loss, accuracy)
def train_baseline(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        outputs = outputs.reshape((-1, outputs.shape[-1])) # 전부 column으로 reshape
        targets = targets.reshape((-1,)) # 전부 column으로 reshape
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


# 모델 예측값과 실제 결과값 비교 -> 결과 출력(loss, accuracy)
def evaluate_baseline(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        outputs = outputs.reshape((-1, outputs.shape[-1]))
        targets = targets.reshape((-1,))
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


# 모델 예측값과 실제 결과값 비교 -> 학습(optimizer step, scheduler step) -> 결과 출력(loss, accuracy)
def train_prototype(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for support_inputs, query_inputs, query_targets in loader:
        support_inputs = support_inputs.to(device)
        query_inputs = query_inputs.to(device)
        query_targets = query_targets.to(device)
        
        query_outputs = model(support_inputs, query_inputs)
        query_outputs = query_outputs.reshape((-1, query_outputs.shape[-1]))
        query_targets = query_targets.reshape((-1,))
        loss = loss_fn(query_outputs, query_targets)
        metric = metric_fn(query_outputs, query_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary


# 모델 예측값과 실제 결과값 비교 -> 결과 출력(loss, accuracy)
def evaluate_prototype(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    
    for support_inputs, query_inputs, query_targets in loader:
        support_inputs = support_inputs.to(device)
        query_inputs = query_inputs.to(device)
        query_targets = query_targets.to(device)
        
        query_outputs = model(support_inputs, query_inputs)
        query_outputs = query_outputs.reshape((-1, query_outputs.shape[-1]))
        query_targets = query_targets.reshape((-1,))
        loss = loss_fn(query_outputs, query_targets)
        metric = metric_fn(query_outputs, query_targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary