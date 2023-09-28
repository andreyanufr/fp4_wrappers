import torch


class FP4CompressDecompress:
    def __init__(self, quants) -> None:
        self.vals = []
        for i in range(len(quants) - 1):
            self.vals.append(0.5 * (quants[i] + quants[i + 1]))

        self.quants = quants

    def dequantize(self, idx):
        return self.quants[idx]

    def quantize(self, x):
        if x > self.vals[7]:
            if x > self.vals[11]:  # 1
                if x > self.vals[13]:  # 11
                    if x > self.vals[14]:  # 111
                        return 0b1111
                    else:
                        return 0b1110
                else:
                    if x > self.vals[12]:  # 110
                        return 0b1101
                    else:
                        return 0b1100
            else:
                if x > self.vals[9]:  # 10
                    if x > self.vals[10]:  # 101
                        return 0b1011
                    else:
                        return 0b1010
                else:
                    if x > self.vals[8]:  # 100
                        return 0b1001
                    else:
                        return 0b1000
        else:
            if x > self.vals[3]:  # 0
                if x > self.vals[5]:  # 01
                    if x > self.vals[6]:  # 011
                        return 0b0111
                    else:
                        return 0b0110
                else:
                    if x > self.vals[4]:  # 010
                        return 0b0101
                    else:
                        return 0b0100
            else:
                if x > self.vals[1]:  # 00
                    if x > self.vals[2]:  # 001
                        return 0b0011
                    else:
                        return 0b0010
                else:
                    if x > self.vals[0]:  # 000
                        return 0b0001
                    else:
                        return 0b0000

    def qd(self, x):
        return self.dequantize(self.quantize(x))


def apply_qd(layer, compressor, exclude, name=""):
    if isinstance(layer, torch.nn.Linear):
        print("FP4 emulation for layer: ", name)
        target_dim = 0
        stat_dim = (target_dim + 1) % 2
        input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
        input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
        scale = torch.maximum(torch.abs(input_high), torch.abs(input_low))
        scale = scale.unsqueeze(stat_dim)

        layer.weight.data = layer.weight.data / scale
        layer.weight.data.apply_(compressor.qd)
        layer.weight.data = layer.weight.data * scale
    else:
        for name, clayer in layer.named_children():
            skip = False
            for nval in exclude:
                if nval in name:
                    skip = True
            if skip:
                continue

            apply_qd(clayer, compressor, exclude, name)


def compression_emulation(
    model, compression_type="nf4", exclude=["lm_head", "logits", "head", "embed_out"]
):
    nf4 = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
    af4 = [
        -1.0,
        -0.69441008,
        -0.51243739,
        -0.3736951,
        -0.25607552,
        -0.14982478,
        -0.04934812,
        0.0,
        0.04273164,
        0.12934483,
        0.21961274,
        0.31675666,
        0.42563882,
        0.55496234,
        0.72424863,
        1.0,
    ]
    pq4 = [
        -1.0,
        -0.7346938775510203,
        -0.5102040816326531,
        -0.32653061224489793,
        -0.18367346938775508,
        -0.08163265306122448,
        -0.02040816326530612,
        0.0,
        0.015625,
        0.0625,
        0.140625,
        0.25,
        0.390625,
        0.5625,
        0.765625,
        1.0,
    ]

    t4 = [
        -1.0,
        -0.676314,
        -0.50419724,
        -0.37671864,
        -0.2702159,
        -0.17508262,
        -0.086152576,
        0.0,
        0.075291716,
        0.152421,
        0.23354281,
        0.3216031,
        0.42128822,
        0.5414709,
        0.7038025,
        1.0,
    ]

    mapping = {"nf4": nf4, "anf4": af4, "pq4": pq4, "t4": t4}

    assert compression_type in mapping

    compressor = FP4CompressDecompress(mapping[compression_type])

    apply_qd(model, compressor, exclude)

    return model
