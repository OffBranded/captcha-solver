import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = '-'
CHAR_LIST = [BLANK] + list(ALPHABET)
INDEX_TO_CHAR = {i: c for i, c in enumerate(CHAR_LIST)}


class SimpleCRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(128*2, n_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(3, 0, 1, 2)  # (width, batch, channels, height)
        x = x.contiguous().view(width, batch, channels*height)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x  # (seq_len, batch, n_classes)

def decode_predictions(output):
    # output shape: (seq_len, batch, n_classes)
    output = output.softmax(2)  # probabilities
    max_probs, max_indices = output.max(2)  # (seq_len, batch)
    max_indices = max_indices.squeeze(1).cpu().numpy()  # (seq_len,)

    # CTC decoding (remove duplicates and blanks)
    decoded = []
    prev_idx = -1
    for idx in max_indices:
        if idx != 0 and idx != prev_idx:  # skip blank (0) and duplicates
            decoded.append(INDEX_TO_CHAR[idx])
        prev_idx = idx
    return ''.join(decoded)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "./model/captcha_model2.pth"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return
    
    n_classes = len(CHAR_LIST)
    model = SimpleCRNN(n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((50, 250)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    folder = "test_data"
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    print(f"Starting prediction on {len(files)} images...\n")
    
    with torch.no_grad():
        for i, filename in enumerate(files, 1):
            path = os.path.join(folder, filename)
            image = Image.open(path).convert('L')
            image = transform(image).unsqueeze(0).to(device)  # (1,1,50,250)
            
            output = model(image)  # (seq_len, batch=1, n_classes)
            
            print(f"Processing [{i}/{len(files)}]: {filename}")
            print(f"Raw output shape (seq_len, batch, classes): {output.shape}")
            
            pred_text = decode_predictions(output)
            print(f"Decoded prediction: {pred_text}\n")

if __name__ == "__main__":
    main()
