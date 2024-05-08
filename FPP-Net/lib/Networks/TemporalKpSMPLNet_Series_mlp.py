import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y


class ContNetwork(nn.Module):
    def __init__(
            self, args,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
    ):
        super(ContNetwork, self).__init__()

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.jointEncoder = nn.Sequential(
            # nn.ReplicationPad1d(1),
            nn.Conv1d(26, 256, kernel_size=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv1d(256, 512, kernel_size=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Conv1d(1024, 2048, kernel_size=1, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )

        # pressure decoder
        self.press_decode = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(),  # relu
            nn.Linear(1024, 1024),
            nn.Dropout(),  # relu
            nn.Linear(1024, 484)
        )
        self.press_output = nn.Sequential(
            nn.Linear(args.seqlen, 1),
            nn.Sigmoid(),
        )

        self.cont_output = nn.Sequential(
            nn.Linear(484, 484),
            nn.Dropout(),  # relu
            nn.Linear(484, 484),
            nn.Dropout(),  # relu
            nn.Linear(484, 192),
            nn.Sigmoid(),
        )

    def forward(self, keypoints):
        # input image feature extractor
        batch_size, seqlen, h, w = keypoints.shape
        keypoints = keypoints.reshape(-1, h, w)
        kp_feat = self.jointEncoder(keypoints)
        kp_feat = kp_feat.reshape(batch_size, seqlen, -1)

        # temporal
        feature = self.encoder(kp_feat)

        # pressure decode
        feature = self.press_decode(feature)  # B, 5, 484
        feature = feature.permute(0, 2, 1)  # B, 484, 5
        xp = self.press_output(feature)  # B, 484, 1
        press_output = xp.reshape(batch_size, -1)  # B, 484

        # contact decode
        cont_output = self.cont_output(press_output)  # B, 484

        return press_output, cont_output

