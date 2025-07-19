# NexTGPT - TikTok Intelligence API

Advanced multimodal AI for TikTok/Reels content analysis.

## Features

- Video Analysis with temporal attention
- Audio processing with MFCC features  
- Text understanding and sentiment analysis
- Multimodal fusion for virality prediction
- Production-ready FastAPI server
- Docker deployment for RunPod

## Quick Start

### RunPod Deployment

1. Create RunPod account
2. Use Docker image: your-username/nextgpt-tiktok:latest
3. Set container disk: 20GB
4. Expose port: 8000
5. Deploy with A100 or RTX 4090 GPU

### API Usage

```python
import requests

with open('video.mp4', 'rb') as f:
    response = requests.post(
        'https://your-endpoint.com/predict',
        files={'video': f},
        data={'text': 'Amazing video! #viral'}
    )

result = response.json()
print(f"Virality Score: {result['predictions']['virality']['score']}")
```

## API Endpoints

- POST /predict - Analyze video
- GET /health - Health check
- GET / - API info

## Response Format

```json
{
  "success": true,
  "predictions": {
    "virality": {"score": 0.753, "category": "high"},
    "quality": {"score": 0.821, "category": "excellent"},
    "engagement": {
      "like_rate": 0.0654,
      "comment_rate": 0.0089
    }
  }
}
```

## License

MIT License
