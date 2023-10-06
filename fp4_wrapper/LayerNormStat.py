import torch


layer_norm_stat = {}


# def getActivation(name):
#     # the hook signature
#     def hook(model, input, output):
#         if hasattr(input, 'logits'):
#             data = input.logit
#         else:
#             data = input
#         dim = [0] if len(data[0].shape) == 2 else [0, 1]
#         if name not in layer_norm_stat:
#             tmp = torch.mean(input[0].detach(), dim=dim).cpu()
#             layer_norm_stat[name] = [1, [tmp]]
#         else:
#             tmp = torch.mean(input[0].detach(), dim=dim).cpu()
#             # tmp = (layer_norm_stat[name][1] * layer_norm_stat[name][0] + tmp) / (
#             #     layer_norm_stat[name][0] + 1
#             # )
#             layer_norm_stat[name][0] += 1
#             layer_norm_stat[name][1].append(tmp)

#     return hook

def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        if hasattr(input[0], 'logits'):
            data = input[0].logits
        else:
            data = input[0]
        dim = [0] if len(data.shape) == 2 else [0, 1]
        if name not in layer_norm_stat:
            tmp = torch.mean(data.detach(), dim=dim).cpu()
            layer_norm_stat[name] = [1, [tmp]]
        else:
            tmp = torch.mean(data.detach(), dim=dim).cpu()
            # tmp = (layer_norm_stat[name][1] * layer_norm_stat[name][0] + tmp) / (
            #     layer_norm_stat[name][0] + 1
            # )
            layer_norm_stat[name][0] += 1
            layer_norm_stat[name][1].append(tmp)

    return hook

class Normalizer(torch.nn.Module):
    def __init__(self, n_feaures):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(n_feaures))
        self.beta = torch.nn.Parameter(torch.zeros(n_feaures))
    
    def forward(self, x):
        if hasattr(x, 'logits'):
            x.logits = x.logits *  self.alpha + self.beta
            return x

        return x * self.alpha + self.beta


class NormalizerWrapper(torch.nn.Module):
    def __init__(self, module, n_features) -> None:
        super().__init__()
        self.module = module
        device = next(module.parameters()).device
        self.norm = Normalizer(n_features).to(device)

    def forward(self , *args, **kwargs):
        y = self.module(*args, **kwargs)
        n_y = self.norm(y)
        return n_y


# def regiset_stat_hooks(hooks, layer, name=""):
#     if isinstance(layer, torch.nn.LayerNorm):
#         h = layer.register_forward_hook(getActivation(name))
#         hooks.append(h)
#         print('Hook for layer: ', name)
#     else:
#         for cname, clayer in layer.named_children():
#             regiset_stat_hooks(hooks, clayer, name + "." + cname)

def regiset_stat_hooks(hooks, layer, name=""):
    if isinstance(layer, Normalizer):
        h = layer.register_forward_hook(getActivation(name))
        hooks.append(h)
        print('Hook for layer: ', name)
    else:
        for cname, clayer in layer.named_children():
            regiset_stat_hooks(hooks, clayer, name + "." + cname)


def ln_forward(ln, x):
    m = torch.mean(x, dim=-1, keepdim=True)
    v = torch.var(x, dim=-1, unbiased=False, keepdim=True)
    x = (x - m) / torch.sqrt(v + ln.eps)
    
    res = x * ln.weight + ln.bias
    return res


def ln_normed(ln, x):
    m = torch.mean(x, dim=-1, keepdim=True)
    v = torch.var(x, dim=-1, unbiased=False, keepdim=True)
    x = (x - m) / torch.sqrt(v + ln.eps)

    return x


def fix_layer_norm(layer, fp32_stats, quant_stats, name="", names=[]):
    if isinstance(layer, torch.nn.LayerNorm) and ('final_layer_norm' in name or '.ln_f' in name):# and 'model.decoder.layers.0.final_layer_norm' in name:
        device = layer.weight.device
        layer.requires_grad_ = False
        layer.weight.requires_grad_ = False
        layer.bias.requires_grad_ = False

        x_fp32 = torch.stack(fp32_stats[name][1], dim=0).to(device)
        x_q = torch.stack(quant_stats[name][1], dim=0).to(device)
        x_fp32 = x_fp32.to(x_q.dtype)
        test = False
        print('Fix layer norm: ', name)
        names.append(name)
        if test:
            y_fp32 = layer(x_fp32)
            y_q = layer(x_q)

            tmp = ln_forward(layer, x_fp32)
            diff = torch.mean(torch.abs(tmp - y_fp32))
            print(diff)
        else:
            y_fp32 = layer(x_fp32)
            x_q = ln_normed(layer, x_q)
            #y_q = layer(x_q)
            # LN(x_q) = y_fp32
            for i in range(x_q.shape[1]): # shape is [seq_len, data_dim]
                A = x_q[:, i]
                ones = torch.ones_like(A)
                A = torch.stack((A, ones), dim=1).to(torch.float32)
                B = y_fp32[:, i].unsqueeze(1).to(torch.float32)
                gamma_beta = torch.linalg.lstsq(A, B)[0].to(x_q.dtype)
                layer.weight.data[i] = gamma_beta[0]
                layer.bias.data[i] = gamma_beta[1]
    else:
        for cname, clayer in layer.named_children():
            fix_layer_norm(clayer, fp32_stats, quant_stats, name + "." + cname, names)
            if len(names) > 0:
                return


def recalibrate_normalizer(layer, fp32_stats, quant_stats, name=""):
    if isinstance(layer, Normalizer):
        device = layer.alpha.device
        layer.requires_grad_ = False
        layer.alpha.requires_grad_ = False
        layer.beta.requires_grad_ = False

        x_fp32 = torch.stack(fp32_stats[name][1], dim=0).to(device)
        x_q = torch.stack(quant_stats[name][1], dim=0).to(device)
        x_fp32 = x_fp32.to(x_q.dtype)
        test = False
        print('Fix layer norm: ', name)
        if test:
            y_fp32 = layer(x_fp32)
            y_q = layer(x_q)

            diff = torch.mean(torch.abs(y_q - y_fp32))
            print(diff)
        else:
            use_pca = True
            if use_pca:
                y_fp32 = layer(x_fp32)
                m = torch.mean(y_fp32, dim=0)
                y_fp32 -= m

                y_q = layer(x_q)
                x_q -= m
                
                diff_before = torch.mean(torch.abs(y_fp32 - y_q))
                pca_dim = 128
                U, S, V = torch.torch.pca_lowrank(y_fp32, q=pca_dim)
                y_fp32_proj = torch.matmul(y_fp32, V)
                x_q_proj = torch.matmul(x_q, V)
                alpha_proj = torch.ones((1, pca_dim))
                beta_proj  = torch.zeros((1, pca_dim))
                for i in range(x_q_proj.shape[1]): # shape is [seq_len, data_dim]
                    A = x_q_proj[:, i]
                    ones = torch.ones_like(A)
                    A = torch.stack((A, ones), dim=1).to(torch.float32).to('cpu')
                    B = y_fp32_proj[:, i].unsqueeze(1).to(torch.float32).to('cpu')
                    alpha_beta = torch.linalg.lstsq(A, B, driver='gelss')[0].to(x_q.dtype)
                    alpha_proj.data[0, i] = alpha_beta[0]
                    beta_proj.data[0, i] = alpha_beta[1]
                alpha = torch.matmul(alpha_proj.to(V.device), torch.t(V))
                beta = torch.matmul(beta_proj.to(V.device), torch.t(V))

                layer.alpha.data[:] = alpha[:]
                layer.beta.data[:] = beta[:]

                y_q = layer(x_q)
                diff_after = torch.mean(torch.abs(y_fp32 - y_q))
                print(diff_before, diff_after)
            else:
                y_fp32 = layer(x_fp32)
                y_q = layer(x_q)
                diff_before = torch.mean(torch.abs(y_fp32 - y_q))
                
                for i in range(x_q.shape[1]): # shape is [seq_len, data_dim]
                    A = x_q[:, i]
                    ones = torch.ones_like(A)
                    A = torch.stack((A, ones), dim=1).to(torch.float32).to('cpu')
                    B = y_fp32[:, i].unsqueeze(1).to(torch.float32).to('cpu')
                    alpha_beta = torch.linalg.lstsq(A, B, driver='gelss')[0].to(x_q.dtype)
                    layer.alpha.data[i] = alpha_beta[0]
                    layer.beta.data[i] = alpha_beta[1]
                y_q = layer(x_q)
                diff_after = torch.mean(torch.abs(y_fp32 - y_q))
                print(diff_before, diff_after)
    else:
        for cname, clayer in layer.named_children():
            recalibrate_normalizer(clayer, fp32_stats, quant_stats, name + "." + cname)



def forward_loop(model, dataset, tokenizer, num_iters, seq_len):
    global layer_norm_stat
    layer_norm_stat = {}
    hooks = []
    regiset_stat_hooks(hooks, model, "")
    device = next(model.parameters()).device

    for i, item in enumerate(dataset):
        if i >= num_iters:
            break
        if len(item["text"]) < 128:
            num_iters += 1
            continue

        text_ids = tokenizer(item["text"])
        input_ids = text_ids['input_ids']
        attn_mask = text_ids['attention_mask']
        
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            attn_mask = attn_mask[:seq_len]

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        attn_mask = torch.tensor(attn_mask).unsqueeze(0).to(device)

        model(input_ids=input_ids, attention_mask=attn_mask)

    for h in hooks:
        h.remove()

    res = dict(layer_norm_stat)
    layer_norm_stat = {}
    return res


