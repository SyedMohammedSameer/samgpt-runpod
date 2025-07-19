
import os
import torch
import tempfile
import logging
import runpod
from model import NexTGPTConfig, NexTGPTInference
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

inference_engine = None

def load_model():
    global inference_engine
    try:
        logger.info("Loading NexTGPT model...")
        config = NexTGPTConfig()
        checkpoint_path = "/app/model/checkpoint.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        inference_engine = NexTGPTInference(checkpoint_path, config)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def download_file(url, dest_path):
    """Download file from URL to destination path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise

def handler(event):
    """RunPod serverless handler function"""
    if not inference_engine:
        return {"error": "Model not loaded"}
    
    input_data = event.get('input', {})
    video_url = input_data.get('video_url')
    text = input_data.get('text', "")
    metadata = input_data.get('metadata', {})
    
    if not video_url:
        return {"error": "No video_url provided"}
    
    temp_video_path = None
    try:
        # Download video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            download_file(video_url, temp_video.name)
            temp_video_path = temp_video.name
        
        # Run inference
        results = inference_engine.predict(
            video_path=temp_video_path,
            text=text,
            metadata=metadata
        )
        
        # Format predictions for response
        formatted_predictions = {
            'virality': {
                'score': round(results['virality_score'], 3),
                'category': 'high' if results['virality_score'] > 0.7 else 'medium' if results['virality_score'] > 0.4 else 'low'
            },
            'quality': {
                'score': round(results['quality_score'], 3),
                'category': 'excellent' if results['quality_score'] > 0.8 else 'good' if results['quality_score'] > 0.6 else 'average'
            },
            'sentiment': {
                'score': round(results['sentiment_score'], 3),
                'category': 'positive' if results['sentiment_score'] > 0.6 else 'neutral' if results['sentiment_score'] > 0.4 else 'negative'
            },
            'engagement': {
                'like_rate': round(results['engagement_prediction']['like_rate'], 4),
                'comment_rate': round(results['engagement_prediction']['comment_rate'], 4),
                'save_rate': round(results['engagement_prediction']['save_rate'], 4)
            },
            'predicted_metrics': {
                'expected_likes': results['predicted_metrics']['expected_likes'],
                'expected_comments': results['predicted_metrics']['expected_comments'],
                'expected_saves': results['predicted_metrics']['expected_saves']
            }
        }
        
        return {
            "success": True,
            "predictions": formatted_predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

if __name__ == "__main__":  # Fixed the syntax error here
    load_model()
    runpod.serverless.start({"handler": handler})
