def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, patience=5, log_fn=None):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.flatten(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        if log_fn:
            log_fn(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

        # 验证
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            if log_fn:
                log_fn("Validation loss improved.")
        else:
            patience_counter += 1
            if log_fn:
                log_fn(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            if log_fn:
                log_fn("Early stopping.")
            break

    return best_model_state

