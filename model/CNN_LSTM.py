import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        enc_len,
        fore_len,
        s_channels,
        img_height,
        img_width,
        hidden_dim,
    ):
        super(CNN_LSTM, self).__init__()
        self.enc_len = enc_len
        self.fore_len = fore_len
        self.output_len = enc_len + fore_len - 1

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(s_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_s = torch.zeros(1, s_channels, img_height, img_width)
            s_feat_dim = self.cnn_encoder(dummy_s).shape[1]

        self.frame_dim = img_height * img_width
        self.hidden_dim = hidden_dim
        self.s_feat_dim = s_feat_dim

        self.lstm_cell = nn.LSTMCell(input_size=self.frame_dim, hidden_size=hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_dim + s_feat_dim, 32 * (img_height // 4) * (img_width // 4)
            ),
            nn.ReLU(),
        )

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x, s, m):
        x = x[:, : self.enc_len]
        batch_size, total_len, _, H, W = x.shape
        device = x.device

        s_feat = self.cnn_encoder(s)

        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros_like(hx)

        outputs = []

        for t in range(self.enc_len):
            x_t = x[:, t, :, :, :].view(batch_size, -1)
            hx, cx = self.lstm_cell(x_t, (hx, cx))
            combined = torch.cat([hx, s_feat], dim=1)
            feature = self.decoder(combined)
            feature = feature.view(batch_size, 32, H // 4, W // 4)
            decoded = self.cnn_decoder(feature)
            outputs.append(decoded.unsqueeze(1))

        last_input = x[:, self.enc_len - 1, :, :, :].view(batch_size, -1)
        for _ in range(self.fore_len - 1):
            hx, cx = self.lstm_cell(last_input, (hx, cx))
            combined = torch.cat([hx, s_feat], dim=1)
            feature = self.decoder(combined)
            feature = feature.view(batch_size, 32, H // 4, W // 4)
            decoded = self.cnn_decoder(feature)
            outputs.append(decoded.unsqueeze(1))
            last_input = decoded.view(batch_size, -1).detach()

        return torch.cat(outputs, dim=1)  # (batch_size, output_len, 1, H, W)
