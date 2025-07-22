import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = '-'  # For CTC blank
CHAR_LIST = [BLANK] + list(ALPHABET)
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHAR_LIST)}

class CaptchaDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(os.path.join(folder, "*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        label_str = os.path.splitext(os.path.basename(path))[0].split('_')[0]  #ABC123_1.jpg
        label_str = label_str.upper()
        label = torch.tensor([CHAR_TO_INDEX[c] for c in label_str], dtype=torch.long)
        
        image = Image.open(path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label, len(label)

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
        # Calculate CNN output width and height after conv+pool
        # Example input: (1, 50, 250)
        # After 2x maxpool: height 50->12, width 250->62 (approx)
        # Flatten height and channels -> sequence length = width, feature size = height * channels
        self.rnn = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(128*2, n_classes)
    
    def forward(self, x):
        # x: (batch, 1, 50, 250)
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(3, 0, 1, 2)  # (width, batch, channels, height)
        x = x.contiguous().view(width, batch, channels*height)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x  # (seq_len, batch, n_classes)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((50, 250)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = CaptchaDataset("./data/originals", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    n_classes = len(CHAR_LIST)
    model = SimpleCRNN(n_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels, label_lengths in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # (seq_len, batch, n_classes)
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
            
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "captcha_model.pth")
    print("Training finished and model saved.")

# Collate function for variable-length labels
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

if __name__ == "__main__":
    train()
