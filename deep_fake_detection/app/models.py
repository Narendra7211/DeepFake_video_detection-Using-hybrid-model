import torch
import torch.nn as nn
import torchvision.models as models

# Define the ResNeXt + LSTM model
class ResNeXtLSTM(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(ResNeXtLSTM, self).__init__()
        model = models.resnext50_32x4d(pretrained = True) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

# Define the EfficientNet + LSTM model
class EfficientNetLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, lstm_layers=1, bidirectional=False):
        super(EfficientNetLSTM, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(1280, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.efficientnet(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        embeddings = torch.mean(lstm_out, dim=1)
        logits = self.fc(embeddings)
        return embeddings, logits


# Define the MobileNet + GRU model
class MobileNetGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, gru_layers=1, bidirectional=False):
        super(MobileNetGRU, self).__init__()

        # Load pretrained MobileNetV3-Large and use its features
        mobilenet = models.mobilenet_v3_large(pretrained=False)  # pretrained=False for deployment
        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch*time, 960, 1, 1)

        # GRU for temporal learning
        self.gru = nn.GRU(
            input_size=960,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Fully connected layer for classification
        gru_output_size = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, channels, height, width)
        Returns:
            embeddings: temporal features (batch_size, gru_output_size)
            logits: classification scores (batch_size, num_classes)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)                  # (batch*time, 3, H, W)
        x = self.feature_extractor(x)              # Extract spatial features
        x = self.avgpool(x)                        # (batch*time, 960, 1, 1)
        x = x.view(b, t, -1)                        # (batch, time, 960)

        gru_out, _ = self.gru(x)                    # (batch, time, hidden_dim)
        embeddings = torch.mean(gru_out, dim=1)     # Temporal average pooling

        logits = self.fc(embeddings)                # (batch, num_classes)
        return embeddings, logits


# Define the EfficientNet + GRU model
class EfficientNetGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, gru_layers=1, bidirectional=False):
        super(EfficientNetGRU, self).__init__()
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.feature_extractor = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(1280, hidden_dim, gru_layers, bidirectional=bidirectional, batch_first=True)
        gru_output_size = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x)
        embeddings = torch.mean(gru_out, dim=1)
        logits = self.fc(embeddings)
        return embeddings, logits


# Define the ResNeXt + GRU model
class ResNeXtGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, gru_layers=1, bidirectional=False):
        super(ResNeXtGRU, self).__init__()
        resnext = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gru = nn.GRU(2048, hidden_dim, gru_layers, bidirectional=bidirectional, batch_first=True)
        gru_output_size = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)
        gru_out, _ = self.gru(x)
        embeddings = torch.mean(gru_out, dim=1)
        logits = self.fc(embeddings)
        return embeddings, logits


# Define the MobileNet + LSTM model
class MobileNetLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, lstm_layers=1, bidirectional=False):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(1280, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.mobilenet(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        embeddings = torch.mean(lstm_out, dim=1)
        logits = self.fc(embeddings)
        return embeddings, logits