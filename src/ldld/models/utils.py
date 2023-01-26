def print_model_summary(model):
    param_cnt = sum(param.numel()
                    for param in model.parameters() if param.requires_grad)

    print(model)
    print(f"Model has {param_cnt:,} trainable parameters")
