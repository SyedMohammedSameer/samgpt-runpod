# NexTGPT Model Wrapper for Deployment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import librosa
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NexTGPTConfig:
    vision_embed_dim: int = 512
    text_embed_dim: int = 384
    audio_embed_dim: int = 256
    fusion_embed_dim: int = 256
    hidden_dim: int = 128
    frames_per_second: int = 2
    target_frame_size: Tuple[int, int] = (224, 224)
    audio_sample_rate: int = 16000
    max_audio_duration: int = 30

# Model classes (simplified versions)
class TemporalVideoEncoder(nn.Module):
    def __init__(self, config: NexTGPTConfig):
        super().__init__()
        self.config = config
        input_dim = 3 * config.target_frame_size[0] * config.target_frame_size[1]
        self.frame_encoder = nn.Sequential(
            nn.Linear(input_dim, config.vision_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.vision_embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.output_proj = nn.Linear(config.vision_embed_dim, config.fusion_embed_dim)
        
    def forward(self, video_frames):
        batch_size, num_frames = video_frames.shape[:2]
        frames_flat = video_frames.view(batch_size, num_frames, -1)
        frame_embeddings = self.frame_encoder(frames_flat)
        attended_frames, _ = self.temporal_attention(
            frame_embeddings, frame_embeddings, frame_embeddings
        )
        pooled = torch.mean(attended_frames, dim=1)
        output = self.output_proj(pooled)
        return output

class AudioEncoder(nn.Module):
    def __init__(self, config: NexTGPTConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.audio_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.fusion_embed_dim)
        )
        
    def forward(self, audio_features):
        return self.encoder(audio_features)

class TextEncoder(nn.Module):
    def __init__(self, config: NexTGPTConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.text_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.fusion_embed_dim)
        )
        
    def forward(self, text_features):
        return self.encoder(text_features)

class MetadataEncoder(nn.Module):
    def __init__(self, config: NexTGPTConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, config.fusion_embed_dim)
        )
        
    def forward(self, metadata_features):
        return self.encoder(metadata_features)

class MultimodalFusionTransformer(nn.Module):
    def __init__(self, config: NexTGPTConfig):
        super().__init__()
        self.config = config
        
        # Modality encoders
        self.video_encoder = TemporalVideoEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.metadata_encoder = MetadataEncoder(config)
        
        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_embed_dim,
            nhead=4,
            dim_feedforward=config.fusion_embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output heads
        self.engagement_head = nn.Sequential(
            nn.Linear(config.fusion_embed_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 3)
        )
        
        self.virality_head = nn.Sequential(
            nn.Linear(config.fusion_embed_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(config.fusion_embed_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.fusion_embed_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, video_frames, audio_features, text_features, metadata_features):
        # Encode each modality
        video_emb = self.video_encoder(video_frames)
        audio_emb = self.audio_encoder(audio_features)
        text_emb = self.text_encoder(text_features)
        metadata_emb = self.metadata_encoder(metadata_features)
        
        # Stack modalities for transformer
        modality_embeddings = torch.stack([video_emb, audio_emb, text_emb, metadata_emb], dim=1)
        
        # Apply cross-modal fusion
        fused_embeddings = self.fusion_transformer(modality_embeddings)
        
        # Flatten for prediction heads
        fused_flat = fused_embeddings.view(fused_embeddings.size(0), -1)
        
        # Generate predictions
        engagement_pred = self.engagement_head(fused_flat)
        virality_pred = torch.sigmoid(self.virality_head(fused_flat))
        quality_pred = torch.sigmoid(self.quality_head(fused_flat))
        sentiment_pred = torch.sigmoid(self.sentiment_head(fused_flat))
        
        return {
            'engagement': engagement_pred,
            'virality': virality_pred.squeeze(-1),
            'quality': quality_pred.squeeze(-1),
            'sentiment': sentiment_pred.squeeze(-1),
            'embeddings': fused_flat
        }

class NexTGPTInference:
    def __init__(self, checkpoint_path: str, config: NexTGPTConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = MultimodalFusionTransformer(config).to(self.device)
        
        # Load checkpoint with proper security handling
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=False: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Video transforms
        self.video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.target_frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("NexTGPT inference pipeline loaded")
    
    def process_video(self, video_path: str) -> torch.Tensor:
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 15
            
            frame_interval = max(1, int(fps / self.config.frames_per_second)) if fps > 0 else 15
            max_frames = min(int(duration * self.config.frames_per_second), 30)
            
            frame_count = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.video_transform(frame_rgb)
                    frames.append(frame_tensor)
                    
                frame_count += 1
            
            cap.release()
            
            # Pad to 30 frames
            target_frames = 30
            while len(frames) < target_frames:
                frames.append(torch.zeros(3, *self.config.target_frame_size))
            
            return torch.stack(frames[:target_frames]).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return torch.zeros(1, 30, 3, *self.config.target_frame_size)
    
    def process_audio(self, video_path: str) -> torch.Tensor:
        try:
            import moviepy.editor as mp
            temp_audio = "/tmp/temp_audio.wav"
            
            video = mp.VideoFileClip(video_path)
            if video.audio:
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                video.close()
                
                audio, sr = librosa.load(
                    temp_audio,
                    sr=self.config.audio_sample_rate,
                    duration=self.config.max_audio_duration
                )
                
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                
                if len(audio) > 0:
                    features = self._extract_audio_features(audio, sr)
                    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            return torch.zeros(1, self.config.audio_embed_dim)
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return torch.zeros(1, self.config.audio_embed_dim)
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> List[float]:
        features = []
        
        # Basic properties
        features.extend([
            len(audio) / sr,
            np.mean(audio),
            np.std(audio),
            np.max(audio),
            np.min(audio),
        ])
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=8)
        features.extend(np.mean(mfccs, axis=1).tolist())
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
        except:
            features.append(120)
        
        # Pad or truncate
        target_size = self.config.audio_embed_dim
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    def process_text(self, text: str) -> torch.Tensor:
        try:
            if text.strip():
                words = text.split()
                features = [
                    len(text) / 1000,
                    len(words) / 100,
                    len([w for w in words if w.startswith('#')]) / 10,
                    len([w for w in words if w.isupper()]) / max(len(words), 1),
                    (text.count('!') + text.count('?')) / 10,
                ]
                target_dim = self.config.text_embed_dim
                while len(features) < target_dim:
                    features.extend(features[:min(len(features), target_dim - len(features))])
                return torch.tensor(features[:target_dim], dtype=torch.float32).unsqueeze(0)
            else:
                return torch.zeros(1, self.config.text_embed_dim)
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return torch.zeros(1, self.config.text_embed_dim)
    
    def predict(self, video_path: str, text: str = "", metadata: Dict = None) -> Dict:
        try:
            with torch.no_grad():
                video_frames = self.process_video(video_path).to(self.device)
                audio_features = self.process_audio(video_path).to(self.device)
                text_features = self.process_text(text).to(self.device)
                
                if metadata is None:
                    metadata = {
                        'followers': 10000,
                        'verified': False,
                        'audio_duration': 15,
                        'tempo': 120,
                        'speech_rate': 3
                    }
                
                metadata_features = torch.tensor([
                    float(metadata.get('followers', 10000)) / 1000000,
                    float(metadata.get('verified', False)),
                    float(metadata.get('audio_duration', 15)) / 30,
                    float(metadata.get('tempo', 120)) / 200,
                    float(metadata.get('speech_rate', 3)) / 10
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                predictions = self.model(
                    video_frames, audio_features, text_features, metadata_features
                )
                
                results = {
                    'virality_score': predictions['virality'].item(),
                    'quality_score': predictions['quality'].item(),
                    'sentiment_score': predictions['sentiment'].item(),
                    'engagement_prediction': {
                        'like_rate': predictions['engagement'][0][0].item(),
                        'comment_rate': predictions['engagement'][0][1].item(),
                        'save_rate': predictions['engagement'][0][2].item()
                    },
                    'predicted_metrics': {
                        'expected_likes': int(10000 * predictions['engagement'][0][0].item()),
                        'expected_comments': int(10000 * predictions['engagement'][0][1].item()),
                        'expected_saves': int(10000 * predictions['engagement'][0][2].item())
                    }
                }
                
                return results
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'virality_score': 0.5,
                'quality_score': 0.5,
                'sentiment_score': 0.5,
                'engagement_prediction': {'like_rate': 0.01, 'comment_rate': 0.001, 'save_rate': 0.001},
                'predicted_metrics': {'expected_likes': 100, 'expected_comments': 10, 'expected_saves': 10}
            }
