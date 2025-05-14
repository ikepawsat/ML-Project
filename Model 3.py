import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import kagglehub


MEL_CHANNELS = 80
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 24000 
MAX_WAV_LENGTH = 10  
MAX_TEXT_LENGTH = 200


class VCTKDataset(Dataset):
    def __init__(self, root_dir=None, max_len=MAX_TEXT_LENGTH):
        if root_dir is None:
            print("Downloading VCTK dataset from Kaggle...")
            self.root_dir = kagglehub.dataset_download("pratt3000/vctk-corpus")
            print(f"Dataset downloaded to {self.root_dir}")
        else:
            self.root_dir = root_dir

        self.max_len = max_len
        self.samples = []
        self.speaker_ids = {}

        wav_dir = os.path.join(self.root_dir, "wav48_silence_trimmed")
        if not os.path.exists(wav_dir):
            wav_dir = os.path.join(self.root_dir, "wav48")  # Alternative path
            if not os.path.exists(wav_dir):   # Search for wav directory
                for root, dirs, files in os.walk(self.root_dir):
                    for dir in dirs:
                        if "wav" in dir.lower():
                            wav_dir = os.path.join(root, dir)
                            break
                    if os.path.exists(wav_dir):
                        break

        txt_dir = os.path.join(self.root_dir, "txt")
        if not os.path.exists(txt_dir):    # Search for txt directory
            for root, dirs, files in os.walk(self.root_dir):
                for dir in dirs:
                    if dir.lower() == "txt":
                        txt_dir = os.path.join(root, dir)
                        break
                if os.path.exists(txt_dir):
                    break

        print(f"Found wav directory: {wav_dir}")
        print(f"Found text directory: {txt_dir}")

        speaker_idx = 0
        for speaker in os.listdir(wav_dir):
            speaker_path = os.path.join(wav_dir, speaker)
            if os.path.isdir(speaker_path):
                if speaker not in self.speaker_ids:
                    self.speaker_ids[speaker] = speaker_idx
                    speaker_idx += 1

                for file in os.listdir(speaker_path):    # Find all wav files for this speaker
                    if file.endswith(".wav") or file.endswith(".flac"):
                        audio_path = os.path.join(speaker_path, file)

                        base_name = os.path.splitext(file)[0]
                        text_path = os.path.join(txt_dir, speaker, f"{base_name}.txt")

                        if os.path.exists(text_path):
                            self.samples.append((audio_path, text_path, speaker))

        print(f"Loaded {len(self.samples)} samples from VCTK corpus")
        print(f"Found {len(self.speaker_ids)} unique speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text_path, speaker = self.samples[idx]

        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)  # Load audio and convert to mel spectrogram

        if len(audio) > MAX_WAV_LENGTH * SAMPLE_RATE:      # Ensure audio is not too long
            audio = audio[:MAX_WAV_LENGTH * SAMPLE_RATE]

        mel_spec = self._get_mel_spectrogram(audio)    # Generate mel spectrogram

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        speaker_id = self.speaker_ids[speaker]

        return {
            'text': text,
            'mel_spectrogram': mel_spec,
            'speaker_id': speaker_id,
            'audio_path': audio_path
        }

    def _get_mel_spectrogram(self, audio):

        stft = librosa.stft(     # Short-time Fourier transform
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT
        )

        magnitude = np.abs(stft)    # Convert to magnitude spectrogram

        mel_basis = librosa.filters.mel(     # Convert to mel scale
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=MEL_CHANNELS
        )
        mel_spec = np.dot(mel_basis, magnitude)
        mel_spec = np.log(np.maximum(mel_spec, 1e-5))

        return torch.FloatTensor(mel_spec)



# TEXT PROCESSING

class TextProcessor:
    def __init__(self, max_len=MAX_TEXT_LENGTH):
        """
        Process text for TTS model
        Args:
            max_len: Maximum text length
        """
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")    # Using a pretrained tokenizer
        self.vocab_size = self.tokenizer.vocab_size

    def process(self, text):
        """Process text to input IDs and attention mask"""

        encoded = self.tokenizer(    # Tokenize text
            text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0]
        }



# ARCHITECTURE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=3, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):    # x: [batch_size, seq_len]
        x = self.token_embedding(x) * np.sqrt(self.token_embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        if mask is not None:
            # Create mask for transformer (1 = keep, 0 = mask)
            src_key_padding_mask = (mask == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)

        return x

class MelDecoder(nn.Module):
    def __init__(self, d_model=512, mel_channels=MEL_CHANNELS, dropout=0.1):
        super(MelDecoder, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        self.decoder_rnn1 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.decoder_rnn2 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )

        self.proj = nn.Linear(d_model, mel_channels)

    def forward(self, encoder_output, mel_targets=None, teacher_forcing_ratio=0.5):
        batch_size = encoder_output.size(0)
        seq_len = encoder_output.size(1)

        # Initialize LSTM hidden states
        h1 = torch.zeros(2, batch_size, encoder_output.size(2)).to(encoder_output.device)
        c1 = torch.zeros(2, batch_size, encoder_output.size(2)).to(encoder_output.device)

        h2 = torch.zeros(1, batch_size, encoder_output.size(2)).to(encoder_output.device)
        c2 = torch.zeros(1, batch_size, encoder_output.size(2)).to(encoder_output.device)

        if mel_targets is None:
            max_len = seq_len * 5  # Text typically expands ~5x in speech

            outputs = torch.zeros(batch_size, max_len, MEL_CHANNELS).to(encoder_output.device)
            decoder_input = torch.zeros(batch_size, 1, encoder_output.size(2)).to(encoder_output.device)

            for t in range(max_len):
                decoder_output1, (h1, c1) = self.decoder_rnn1(decoder_input, (h1, c1))   # First LSTM

                attn_output, _ = self.attention(   # Attention mechanism
                    query=decoder_output1,
                    key=encoder_output,
                    value=encoder_output
                )

                decoder_output2, (h2, c2) = self.decoder_rnn2(attn_output, (h2, c2))  # Second LSTM

                # Project to mel spectrogram
                output = self.proj(decoder_output2)
                outputs[:, t:t+1] = output

                decoder_input = decoder_output2  # Use current output as next input

            return outputs

        else:
            target_len = mel_targets.size(1)

            outputs = torch.zeros(batch_size, target_len, MEL_CHANNELS).to(encoder_output.device)

            decoder_input = torch.zeros(batch_size, 1, encoder_output.size(2)).to(encoder_output.device)   # Initial decoder input is zero

            for t in range(target_len):
                decoder_output1, (h1, c1) = self.decoder_rnn1(decoder_input, (h1, c1))   # First LSTM


                attn_output, _ = self.attention(  # Attention mechanism
                    query=decoder_output1,
                    key=encoder_output,
                    value=encoder_output
                )


                decoder_output2, (h2, c2) = self.decoder_rnn2(attn_output, (h2, c2))   # Second LSTM

                output = self.proj(decoder_output2)   # Project to mel spectrogram
                outputs[:, t:t+1] = output

                if torch.rand(1).item() < teacher_forcing_ratio:
                    decoder_input = decoder_output2  # Use model output
                else:
                    target_proj = nn.Linear(MEL_CHANNELS, encoder_output.size(2)).to(encoder_output.device)
                    decoder_input = target_proj(mel_targets[:, t:t+1])

            return outputs

class SpeakerEncoder(nn.Module):
    def __init__(self, n_speakers, d_model=512):
        super(SpeakerEncoder, self).__init__()
        self.speaker_embedding = nn.Embedding(n_speakers, d_model)

    def forward(self, speaker_ids):   # speaker_ids: [batch_size]
        return self.speaker_embedding(speaker_ids)

class TTSModel(nn.Module):
    def __init__(self, vocab_size, n_speakers, d_model=512):
        super(TTSModel, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model)
        self.speaker_encoder = SpeakerEncoder(n_speakers, d_model)
        self.mel_decoder = MelDecoder(d_model)
        self.d_model = d_model

    def forward(self, input_ids, speaker_ids, attention_mask=None, mel_targets=None, teacher_forcing_ratio=0.5):
        encoder_output = self.text_encoder(input_ids, attention_mask)  # Text encoding

        speaker_embed = self.speaker_encoder(speaker_ids)   # Speaker embedding

        speaker_embed = speaker_embed.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
        encoder_output = encoder_output + speaker_embed

        mel_outputs = self.mel_decoder(encoder_output, mel_targets, teacher_forcing_ratio)

        return mel_outputs




# VOCODER    -     Use a simple Griffin-Lim vocoder for now

def griffin_lim_vocoder(mel_spectrogram, n_iter=32):
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=MEL_CHANNELS)   # Convert to linear scale
    mel_to_linear = np.linalg.pinv(mel_basis)

    mel_spec = mel_spectrogram.numpy()
    mel_spec = np.exp(mel_spec)

    linear_spec = np.dot(mel_to_linear, mel_spec)   # Convert to linear spectrogram

    # Griffin-Lim algorithm
    angles = np.random.random_sample(linear_spec.shape) * 2 * np.pi - np.pi
    angles = np.exp(1.0j * angles)

    stft_reconstruction = linear_spec.astype(np.complex) * angles

    for _ in range(n_iter):
        audio = librosa.istft(stft_reconstruction, hop_length=HOP_LENGTH, win_length=N_FFT)
        stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT)
        angles = np.exp(1.0j * np.angle(stft))
        stft_reconstruction = linear_spec.astype(np.complex) * angles

    audio = librosa.istft(stft_reconstruction, hop_length=HOP_LENGTH, win_length=N_FFT)
    return audio




# TRAINING

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    mel_specs = [item['mel_spectrogram'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]

    return {
        'texts': texts,
        'mel_spectrograms': mel_specs,  # These have different lengths
        'speaker_ids': torch.tensor(speaker_ids),
        'audio_paths': audio_paths
    }

def train(model, data_loader, text_processor, optimizer, criterion, device, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(data_loader):
            texts = batch['texts']
            mel_specs = batch['mel_spectrograms']
            speaker_ids = batch['speaker_ids'].to(device)

            processed_texts = [text_processor.process(text) for text in texts]
            input_ids = torch.stack([item['input_ids'] for item in processed_texts]).to(device)
            attention_masks = torch.stack([item['attention_mask'] for item in processed_texts]).to(device)

            max_mel_len = max(spec.shape[1] for spec in mel_specs)
            padded_mels = []

            for mel in mel_specs:
                # Pad if needed
                if mel.shape[1] < max_mel_len:
                    padding = torch.zeros(MEL_CHANNELS, max_mel_len - mel.shape[1])
                    padded_mel = torch.cat([mel, padding], dim=1)
                else:
                    padded_mel = mel
                padded_mels.append(padded_mel)

            mel_specs_tensor = torch.stack(padded_mels).permute(0, 2, 1).to(device)

            outputs = model(
                input_ids=input_ids,
                speaker_ids=speaker_ids,
                attention_mask=attention_masks,
                mel_targets=mel_specs_tensor,
                teacher_forcing_ratio=0.5
            )

            loss = criterion(outputs, mel_specs_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"tts_model_epoch_{epoch+1}.pt")

# 6. INFERENCE FUNCTION

def inference(model, text, speaker_id, text_processor, device):
    model.eval()

    processed_text = text_processor.process(text)
    input_ids = processed_text['input_ids'].unsqueeze(0).to(device)
    attention_mask = processed_text['attention_mask'].unsqueeze(0).to(device)

    speaker_id_tensor = torch.tensor([speaker_id]).to(device)

    with torch.no_grad():
        mel_output = model(
            input_ids=input_ids,
            speaker_ids=speaker_id_tensor,
            attention_mask=attention_mask
        )

    audio = griffin_lim_vocoder(mel_output[0].cpu())

    return audio, mel_output[0].cpu()

import torch.nn.functional as F

class SpectralLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512], hop_size=256, win_length=None):
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_size = hop_size
        self.win_length = win_length or hop_size

    def stft_mag(self, x, fft_size):
        x = x.transpose(1, 2)  # (batch, mel_dim, time)
        x = x.reshape(x.shape[0], -1)  # Collapse to mono-like signal

        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            return_complex=True
        )
        return torch.abs(stft)

    def forward(self, x, y):
        loss = 0.0
        for fft_size in self.fft_sizes:
            x_mag = self.stft_mag(x, fft_size)
            y_mag = self.stft_mag(y, fft_size)
            loss += F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        return loss / len(self.fft_sizes)


# 7. MAIN FUNCTION

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = VCTKDataset() 

    data_loader = DataLoader(
        dataset,
        batch_size=32, 
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    text_processor = TextProcessor()

    model = TTSModel(
        vocab_size=text_processor.vocab_size,
        n_speakers=len(dataset.speaker_ids),
        d_model=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = SpectralLoss()

    train(model, data_loader, text_processor, optimizer, criterion, device, epochs=10)

    sample_text = "It’s bread, and cinnamon, and frosting."
    sample_speaker_id = 0  # First speaker in the dataset

    audio, mel = inference(model, sample_text, sample_speaker_id, text_processor, device)

    import soundfile as sf
    sf.write("output_speech.wav", audio, SAMPLE_RATE)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Generated Mel Spectrogram')
    plt.tight_layout()
    plt.savefig("output_mel.png")

if __name__ == "__main__":
    main()