def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters of a Huggingface model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def print_special_tokens(tokenizer):
    from tabulate import tabulate

    # print all special tokens
    special_tokens = tokenizer.special_tokens_map.items()
    table = []
    for token_type, token in special_tokens:
        # if the token is a list, print all tokens
        if isinstance(token, list):
            token_ids = tokenizer.convert_tokens_to_ids(token)
            table.extend(
                [token_type, token, token_id]
                for token, token_id in zip(token, token_ids)
            )
        else:
            token_id = tokenizer.convert_tokens_to_ids(token)
            table.append([token_type, token, token_id])
    print(tabulate(table, headers=["Token Type", "Token", "Token ID"], tablefmt="grid"))
