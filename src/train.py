"""
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J Unniversity
"""

import torch
import torch.optim as optim
from nn_arch.vgg16 import VGG16
from config.config import Config
from utils.tokenizer import Tokenizer
from gpu_config.check import check_gpu
from dataset import ImageCaptionDataset
from torch.utils.data import DataLoader
from nn_arch.caption_model import CaptionModel
from utils.caption_processor import CaptionProcessor

def main():
    check_gpu()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature extraction
    extractor = VGG16(device)
    image_dir = Config().TRAIN_IMAGE_PATH
    features = extractor.extract_features(image_dir)
    features = extractor.load_features()
    
    # Caption processing
    processor = CaptionProcessor()
    mapping = processor.load_captions()
    processor.clean_captions()
    all_captions = processor.get_all_captions()
    
    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = tokenizer.get_vocab_size()
    max_length = max(len(caption.split()) for caption in all_captions)
    
    # Data split
    image_ids = [img_id for img_id in mapping.keys() if img_id in features]
    split = int(len(image_ids) * 0.95)
    train = image_ids[:split]
    test = image_ids[split:]
    
    # Dataset and DataLoader
    train_dataset = ImageCaptionDataset(train, mapping, features, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model training
    model = CaptionModel(vocab_size, max_length).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (img_features, captions, targets) in enumerate(train_loader):
            img_features = img_features.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_features, captions)
            
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    torch.save(model.state_dict(), Config().PATH_TO_SAVE_TRAINED_MODEL +'best_model.pt')

if __name__ == "__main__":
    main()