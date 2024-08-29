import torch
import torch.nn as nn
from PhantomNet import PhantomNet
import os
import librosa
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random




class AudioDataset(Dataset):
    def __init__(self, list_dir, sample_rate=16000, fixed_length=16000 * 7, num_samples=None):
        self.list_dir = list_dir
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length
        all_files = os.listdir(list_dir)

        if num_samples is not None:
            self.files = random.sample(all_files, num_samples)
        else:
            self.files = all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.list_dir, self.files[idx])
        waveform, sr = librosa.load(file_path, sr=self.sample_rate)

        if len(waveform) > self.fixed_length:
            waveform = waveform[:self.fixed_length]
        else:
            padding = self.fixed_length - len(waveform)
            waveform = F.pad(torch.Tensor(waveform), (0, padding), "constant")

        return torch.Tensor(waveform), sr


list_dir = 'PhantomNet/DATA/flac_E_eval/'
save_dir = 'PhantomNet/saved_models/'
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhantomNet(feature_size=1920, num_classes=1,conv_projection=False, use_mode='extractor').to(device)
model.load_state_dict(torch.load("PhantomNet/saved_models/PhantomNet_Extractor_Epoch21_KL:1.6071.pt", map_location=device))
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)



def distillation_loss_KL(student_outputs, teacher_outputs, T=1.0):
    student_probs = F.log_softmax(student_outputs / T, dim=-1)
    teacher_probs = F.softmax(teacher_outputs / T, dim=-1)
    kl_div = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
    return kl_div

def random_crop(audio, crop_size):
    if len(audio) >= crop_size:
        start = random.randint(0, len(audio) - crop_size)
        return audio[start:start + crop_size]
    else:
        return audio


def contrastive_loss_normalized(anchor, positive, negative, margin=1.0):
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

teacher_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-2b").to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")
teacher_model.eval()


def extract_teacher_embeddings(audio):
    with torch.no_grad():
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        outputs = teacher_model(**inputs)
        return outputs.last_hidden_state

##############################
#### TRAIN EMBEDDINGS ########

def train_embeddings(model,epochs,optimizer,num_samples):
 for epoch in range (epochs):
  epoch_loss = 0



  files = os.listdir(list_dir)
  for fi in tqdm(files[:num_samples]):
    waveform, sr = librosa.load(os.path.join(list_dir,fi), sr=16000)
    input_teacher = torch.Tensor(waveform).to(device)
    input_student = input_teacher.unsqueeze(0).to(device)
    teacher_embedding = extract_teacher_embeddings(input_teacher)
    student_embedding = model(input_student)
#    print('shape of student_embedding: ',student_embedding.shape)
    loss = distillation_loss_KL(student_embedding,teacher_embedding)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()


  avg_epoch_loss = epoch_loss / len(files)
  print(f"Epoch {epoch + 22} loss: {avg_epoch_loss}")

  ###################
  #####save model####
  model_filename = os.path.join(save_dir, f"PhantomNet_Extractor_Epoch{epoch + 22}_KL:{avg_epoch_loss:.4f}.pt")
  torch.save(model.state_dict(), model_filename)
  print(f"Model saved to {model_filename}")





##########################
### SSL training #########
def train_SSL(model, epochs, optimizer):

    dataset = AudioDataset(list_dir,fixed_length=16000 * 7,num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for waveform, sr in tqdm(dataloader):
            positive = random_crop(waveform, crop_size=16000)
            negative = random_crop(waveform, crop_size=16000)
            anchor = random_crop(waveform, crop_size=16000)

            positive = model(positive.to(device))
            negative = model(negative.to(device))
            anchor = model(anchor.to(device))

            loss = contrastive_loss_normalized(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss}")

        model_filename = os.path.join(save_dir, f"PhantomNet_SSL_Epoch{epoch + 1}_Loss:{avg_epoch_loss:.4f}.pt")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")


train_embeddings(model=model, epochs=30,optimizer=optimizer,num_samples=100000)






