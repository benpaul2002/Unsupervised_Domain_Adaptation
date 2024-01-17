import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from data import MNIST_Dataset, MNISTM_Dataset, USPS_Dataset, SVHN_Dataset, Office31_Dataset, ImageNet_Dataset, Caltech_Dataset, DomainNet_Dataset, ConcatenatedDataset

from models import MNIST_MNISTM, MNIST_USPS, SVHN_MNIST, Office, ImageNet, MNIST_MNISTM_SVHN, DomainNet
from utils import GrayscaleToRgb, PadSize
from revgrad import get_discriminator
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adaptation_combinations = {
  "MNIST": ["MNIST-M", "USPS", "SVHN"],
  "MNIST-M": ["MNIST"],
  "USPS": ["MNIST"],
  "SVHN": ["MNIST"],
  "Amazon": ["Webcam"],
  "Webcam": ["DSLR"],
  "DSLR": ["Amazon"],
  "Quickdraw": ["Clipart"],
  "Sketch": ["Painting"],
  "Painting": ["Sketch"],
  "MNIST_MNIST-M": ["SVHN"],
}

# Train model on MNIST
# Do Domain adaptation from MNIST to MNIST-M
# Do Domain adaptation from this to SVHN

def calculate_label_distribution(dataset, indices):
    # Extract labels from the specified indices in the dataset
    labels = [dataset[idx][1] for idx in indices]

    # Calculate label distribution
    label_distribution = {label: labels.count(label) for label in set(labels)}

    return label_distribution

def plot_label_distribution(train_label_distribution, val_label_distribution):
    # Plot label distribution
    fig, ax = plt.subplots()
    ax.bar(train_label_distribution.keys(), train_label_distribution.values(), label='Train')
    ax.bar(val_label_distribution.keys(), val_label_distribution.values(), label='Validation')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    ax.set_title('Label Distribution')
    ax.legend()
    plt.show()

def create_train_val_loader(batch_size, dataset):
  shuffled_indices = np.random.permutation(len(dataset))
  train_idx = shuffled_indices[:int(0.8*len(dataset))]       
  val_idx = shuffled_indices[int(0.8*len(dataset)):]

  train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx),
                            num_workers=1, pin_memory=True)
  val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=SubsetRandomSampler(val_idx),
                          num_workers=1, pin_memory=True)
  
  # train_label_distribution = calculate_label_distribution(dataset, train_idx)
  # print("Train Label Distribution:")
  # print(train_label_distribution)

  # # Calculate and display label distribution in validation set
  # val_label_distribution = calculate_label_distribution(dataset, val_idx)
  # print("\nValidation Label Distribution:")
  # print(val_label_distribution)

  # plot label distribution
  # plot_label_distribution(train_label_distribution, val_label_distribution)
  
  return train_loader, val_loader

def get_dataset(source, target):
  if source == "MNIST" and target == "MNIST-M":
    source_dataset = MNIST_Dataset(grayscaleToRgb=True)
    target_dataset = MNISTM_Dataset()
    batch_size = 64

  elif source == "MNIST" and target == "USPS":
    source_dataset = MNIST_Dataset()
    target_dataset = USPS_Dataset()
    batch_size = 64

  elif source == "MNIST" and target == "SVHN":
    source_dataset = MNIST_Dataset(grayscaleToRgb=True, padSize=True, image_size=(32, 32))
    target_dataset = SVHN_Dataset()
    batch_size = 64

  elif source == "SVHN" and target == "MNIST":
    source_dataset = SVHN_Dataset()
    target_dataset = MNIST_Dataset(grayscaleToRgb=True, padSize=True, image_size=(32, 32))
    batch_size = 64

  elif source == "Amazon" and target == "Webcam":
    source_dataset = Office31_Dataset(domain=source)
    target_dataset = Office31_Dataset(domain=target)
    batch_size = 8

  elif source == "Webcam" and target == "DSLR":
    source_dataset = Office31_Dataset(domain=source)
    target_dataset = Office31_Dataset(domain=target)
    batch_size = 64

  elif source == "DSLR" and target == "Amazon":
    source_dataset = Office31_Dataset(domain=source)
    target_dataset = Office31_Dataset(domain=target)
    batch_size = 64

  elif source == "Quickdraw" and target == "Clipart":
    source_dataset =  DomainNet_Dataset(domain=source)
    target_dataset = DomainNet_Dataset(domain=target)
    batch_size = 4

  elif source == "Sketch" and target == "Painting":
    source_dataset =  DomainNet_Dataset(domain=source)
    target_dataset = DomainNet_Dataset(domain=target)
    batch_size = 4

  elif source == "Painting" and target == "Sketch":
    source_dataset =  DomainNet_Dataset(domain=source)
    target_dataset = DomainNet_Dataset(domain=target)
    batch_size = 4

  elif source == "MNIST_MNIST-M" and target == "SVHN":
    source_dataset = MNIST_Dataset(grayscaleToRgb=True, padSize=True, image_size=(32, 32))
    target_dataset = SVHN_Dataset()
    batch_size = 64

  else:
    print("Invalid combination of datasets")
    exit()

  return source_dataset, target_dataset, batch_size

def get_data_loaders(source_dataset, target_dataset, batch_size, choice=1):
  half_batch = batch_size // 2

  if choice == 1:
    train_loader, val_loader = create_train_val_loader(batch_size, source_dataset)
    return train_loader, val_loader

  elif choice == 2:
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                                  shuffle=True, num_workers=1, pin_memory=True)
          
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                              shuffle=True, num_workers=1, pin_memory=True)
    return source_loader, target_loader
  
  elif choice == 3:
    test_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=1, pin_memory=True)
    return test_loader

def get_train_data(source, target, bt=64):
  if type(source) == str:
    source_dataset, target_dataset, batch_size = get_dataset(source, target)
  else:
    source_dataset, target_dataset = source, target
    batch_size = bt
  train_loader, val_loader = get_data_loaders(source_dataset, target_dataset, batch_size, choice=1)
  return train_loader, val_loader, batch_size

def get_model(source_dataset, target_dataset):
  if source_dataset == "MNIST" and target_dataset == "MNIST-M":
    model = MNIST_MNISTM().to(device)

  elif source_dataset == "MNIST" and target_dataset == "USPS":
    model = MNIST_USPS().to(device)

  elif source_dataset == "MNIST" and target_dataset == "SVHN":
    model = SVHN_MNIST().to(device)

  elif source_dataset == "SVHN" and target_dataset == "MNIST":
    model = SVHN_MNIST().to(device)

  elif source_dataset == "Amazon" and target_dataset == "Webcam":
    model = Office().to(device)
    # for name, param in model.named_parameters():
    #   if param.requires_grad:
    #       print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")

  elif source_dataset == "Webcam" and target_dataset == "DSLR":
    model = Office().to(device)

  elif source_dataset == "DSLR" and target_dataset == "Amazon":
    model = Office().to(device)

  elif source_dataset == "Quickdraw" and target_dataset == "Clipart":
    model = DomainNet(num_classes=10).to(device)

  elif source_dataset == "Sketch" and target_dataset == "Painting":
    model = DomainNet(num_classes=10).to(device)  

  elif source_dataset == "Painting" and target_dataset == "Sketch":
    model = DomainNet(num_classes=10).to(device)  

  elif source_dataset == "MNIST_MNIST-M" and target_dataset == "SVHN":
    model = MNIST_MNISTM_SVHN().to(device)

  else:
    print("Invalid combination of datasets")
    exit()

  return model

def epoch_fn(model, dataloader, criterion, optim=None):
  total_loss = 0
  total_accuracy = 0
  for x, y_true in tqdm(dataloader, leave=False):
    x, y_true = x.to(device), y_true.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    if optim is not None:
      optim.zero_grad()
      loss.backward()
      optim.step()

    total_loss += loss.item()
    total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
  mean_loss = total_loss / len(dataloader)
  mean_accuracy = total_accuracy / len(dataloader)

  return mean_loss, mean_accuracy

def source_train(model, train_loader, val_loader, epochs=10):
  optim = torch.optim.Adam(model.parameters())
  lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
  criterion = torch.nn.CrossEntropyLoss()

  best_accuracy = 0
  for epoch in range(1, epochs+1):
      model.train()
      train_loss, train_accuracy = epoch_fn(model, train_loader, criterion, optim=optim)

      model.eval()
      with torch.no_grad():
          val_loss, val_accuracy = epoch_fn(model, val_loader, criterion, optim=None)

      tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                  f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

      if val_accuracy > best_accuracy:
          print('Saving model...')
          best_accuracy = val_accuracy
          torch.save(model.state_dict(), 'trained_models/source.pt')

      lr_schedule.step(val_loss)

def half_batch_train(source, target, model, batch_size, model_file, epochs=10, source_loader=None, target_loader=None):
  if type(source) == str:
    source_dataset, target_dataset, _ = get_dataset(source, target)
  else:
    source_dataset, target_dataset = source, target

  if source_loader is None or target_loader is None:
    source_loader, target_loader = get_data_loaders(source_dataset, target_dataset, batch_size, choice=2)

  if type(source) == str:
    discriminator = get_discriminator(source, target)
  else:
    discriminator = get_discriminator("MNIST_MNIST-M", "SVHN")

  optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

  model.load_state_dict(torch.load(model_file))
  feature_extractor = model.feature_extractor
  clf = model.classifier

  for epoch in range(1, epochs+1):
      batches = zip(source_loader, target_loader)
      n_batches = min(len(source_loader), len(target_loader))

      total_domain_loss = total_label_accuracy = 0
      for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
          source_x = source_x.to(device)
          target_x = target_x.to(device)

          # True labels for domain classifier (0 for source, 1 for target)
          source_domain_y = torch.ones(source_x.shape[0]).to(device)
          target_domain_y = torch.zeros(target_x.shape[0]).to(device)
          label_y = source_labels.to(device)

          # Forward pass for source data
          source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
          source_domain_preds = discriminator(source_features).squeeze()
          source_label_preds = clf(source_features)
          
          target_features = feature_extractor(target_x).view(target_x.shape[0], -1)
          target_domain_preds = discriminator(target_features).squeeze()
          target_label_preds = clf(target_features)

          source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_preds, source_domain_y)
          target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_preds, target_domain_y)
          label_loss = F.cross_entropy(source_label_preds, label_y)
          loss = source_domain_loss + target_domain_loss + label_loss

          optim.zero_grad()
          loss.backward()
          optim.step()

          total_domain_loss += source_domain_loss.item() + target_domain_loss.item()
          total_label_accuracy += (source_label_preds.max(1)[1] == label_y).float().mean().item()

      mean_loss = total_domain_loss / n_batches
      mean_accuracy = total_label_accuracy / n_batches
      tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
              f'source_accuracy={mean_accuracy:.4f}')

  torch.save(model.state_dict(), 'trained_models/revgrad.pt')

def test(source, target, model, model_file):
  source_dataset, target_dataset, _ = get_dataset(source, target)

  test_loader = get_data_loaders(source_dataset, target_dataset, batch_size=64, choice=3)
  model.load_state_dict(torch.load(model_file))
  model.eval()

  total_accuracy = 0
  with torch.no_grad():
      for x, y_true in tqdm(test_loader, leave=False):
          x, y_true = x.to(device), y_true.to(device)
          y_pred = model(x)
          total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
  
  mean_accuracy = total_accuracy / len(test_loader)
  print(f'Accuracy on target data: {mean_accuracy:.4f}')

def main():
  print("Domain Adaptation!")

  print("Pick a source dataset: ")
  for i in range(len(adaptation_combinations)):
    print(f"{i+1}. {list(adaptation_combinations.keys())[i]}")
  source_dataset = list(adaptation_combinations.keys())[int(input())-1]
  print()

  print("Pick a target dataset: ")
  for i in range(len(adaptation_combinations[source_dataset])):
    print(f"{i+1}. {adaptation_combinations[source_dataset][i]}")
  target_dataset = adaptation_combinations[source_dataset][int(input())-1]
  print()

  print("Adaptation from", source_dataset, "to", target_dataset)

  epochs = 1

  if source_dataset == "MNIST_MNIST-M" and target_dataset == "SVHN":
    source_dt = MNIST_Dataset(grayscaleToRgb=True, padSize=True, image_size=(32, 32))
    target_dt1 = MNISTM_Dataset(target_size=(32, 32))
    target_dt2 = SVHN_Dataset()

    batch_size = 64
    train_loader, val_loader, _ = get_train_data(source_dt, target_dt1)
    model = get_model(source_dataset, target_dataset)
    source_train(model, train_loader, val_loader, epochs)  

    print("----------------------------------------")
    print("Source training complete")
    print("----------------------------------------")

    source_model_file = 'trained_models/source.pt'
    half_batch_train(source_dt, target_dt1, model, batch_size, source_model_file, epochs)

    print("----------------------------------------")
    print("Half batch training complete")
    print("----------------------------------------")

    revgrad_model_file1 = 'trained_models/revgrad.pt'

    source_target_dt1_concat = ConcatenatedDataset(source_dt, target_dt1)
    concatenated_dataloader = DataLoader(source_target_dt1_concat, batch_size=batch_size, shuffle=True)
    target_dt2_loader = DataLoader(target_dt2, batch_size=batch_size, shuffle=True)
    half_batch_train(source_target_dt1_concat, target_dt2, model, batch_size, revgrad_model_file1, epochs, concatenated_dataloader, target_dt2_loader)

    print("----------------------------------------")
    print("Quarter batch training complete")
    print("----------------------------------------")

    revgrad_model_file2 = 'trained_models/revgrad.pt'
    test(source_dataset, target_dataset, model, revgrad_model_file2)

    return

  train_loader, val_loader, batch_size = get_train_data(source_dataset, target_dataset)
  model = get_model(source_dataset, target_dataset)
  source_train(model, train_loader, val_loader, epochs)

  print("----------------------------------------")
  print("Source training complete")
  print("----------------------------------------")

  source_model_file = 'trained_models/source.pt'
  half_batch_train(source_dataset, target_dataset, model, batch_size, source_model_file, epochs)

  print("----------------------------------------")
  print("Half batch training complete")
  print("----------------------------------------")

  revgrad_model_file = 'trained_models/revgrad.pt'
  test(source_dataset, target_dataset, model, revgrad_model_file)

if __name__ == '__main__':
  main()
