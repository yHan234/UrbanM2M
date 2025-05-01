import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(
        self, enc_len, fore_len, input_size=(64, 64), hidden_dim=64, num_layers=2
    ):
        """
        初始化CNN-LSTM模型

        参数:
            enc_len: 输入序列长度
            fore_len: 预测序列长度
            input_size: 输入图片尺寸 (H, W)
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super(CNN_LSTM, self).__init__()
        self.enc_len = enc_len
        self.fore_len = fore_len
        self.output_len = enc_len + fore_len - 1

        # CNN编码器
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # 计算CNN输出后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            cnn_output_dim = self.cnn_encoder(dummy_input).shape[1]

        # LSTM网络
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 解码器全连接层
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, cnn_output_dim), nn.ReLU())

        # CNN解码器
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),  # 假设输入图片在[0,1]范围内
        )

    def forward(self, x, _=None, __=None):
        """
        前向传播

        参数:
            x: 输入序列 (batch_size, enc_len, 1, H, W)

        返回:
            输出序列 (batch_size, output_len, 1, H, W)
        """
        batch_size = x.size(0)

        # CNN编码
        encoded_frames = []
        for t in range(self.enc_len):
            # 对每个时间步的图片进行编码
            encoded = self.cnn_encoder(x[:, t, :, :, :])  # (batch_size, cnn_output_dim)
            encoded_frames.append(encoded)

        # 堆叠编码后的特征
        encoded_sequence = torch.stack(
            encoded_frames, dim=1
        )  # (batch_size, enc_len, cnn_output_dim)

        # LSTM处理
        lstm_out, _ = self.lstm(encoded_sequence)  # (batch_size, enc_len, hidden_dim)

        # 解码特征
        decoded_features = self.decoder(
            lstm_out
        )  # (batch_size, enc_len, cnn_output_dim)

        # 初始化输出序列
        outputs = []

        # 添加编码部分的输出
        for t in range(self.enc_len):
            # 重塑为CNN解码器期望的形状
            feature = decoded_features[:, t, :]  # (batch_size, cnn_output_dim)

            # 需要知道原始CNN编码后的形状才能正确重塑
            # 这里假设CNN编码后是 (32, H/4, W/4)
            H, W = x.shape[-2], x.shape[-1]
            feature = feature.view(batch_size, 32, H // 4, W // 4)

            # CNN解码
            decoded_img = self.cnn_decoder(feature)  # (batch_size, 1, H, W)
            outputs.append(decoded_img.unsqueeze(1))  # 添加时间维度

        # 预测未来帧
        if self.fore_len > 1:
            # 使用最后一个隐藏状态预测未来
            last_hidden = lstm_out[:, -1:, :]  # (batch_size, 1, hidden_dim)

            for _ in range(self.fore_len - 1):
                # 解码特征
                feature = self.decoder(last_hidden)  # (batch_size, 1, cnn_output_dim)

                # 重塑并解码为图片
                H, W = x.shape[-2], x.shape[-1]
                feature = feature.view(batch_size, 32, H // 4, W // 4)
                decoded_img = self.cnn_decoder(feature)  # (batch_size, 1, H, W)
                outputs.append(decoded_img.unsqueeze(1))

                # 更新LSTM状态 (这里简化处理，实际可能需要更复杂的处理)
                last_hidden, _ = self.lstm(feature.view(batch_size, 1, -1))

        # 合并所有输出
        output_sequence = torch.cat(outputs, dim=1)  # (batch_size, output_len, 1, H, W)

        return output_sequence
