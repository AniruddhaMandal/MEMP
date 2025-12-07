import torch

def train_epoch(model : torch.nn.Module, dataloader, loss_fn, metric_fn, optimizer, scheduler, device):
    total_loss, total_acc = 0, 0
    model.train()
    for i, batch in enumerate(dataloader):
        batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        y = batch.y
        loss = loss_fn(y_pred,y)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Evaluate model before this line 
        acc = metric_fn(y_pred,y)
        total_loss += loss
        total_acc += acc
    if scheduler != None:
        scheduler.step(total_loss)
    avg_loss = total_loss/(i+1)
    avg_acc = total_acc/(i+1)
    return avg_loss, avg_acc

@torch.no_grad()
def test_epoch(model, dataloader, loss_fn, metric_fn, device):
    avg_loss, avg_acc = 0, 0
    model.eval()
    for i, batch in enumerate(dataloader):
        batch.to(device)
        y_pred = model(batch)
        y = batch.y
        loss = loss_fn(y_pred, y)
        acc = metric_fn(y_pred, y)
        avg_loss += loss
        avg_acc += acc
    avg_loss = avg_loss/(i+1)
    avg_acc = avg_acc/(i+1)
    return avg_loss, avg_acc