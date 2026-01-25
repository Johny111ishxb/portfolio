from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from PIL import Image
import io
import base64
from pathlib import Path
import tempfile
import shutil
import logging
from typing import Tuple, Optional, Dict, Any
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AdvancedLaMaInpainter:
    def __init__(self, model_path="big-lama.pt", device="auto"):
        """
        Initialize with .pt model file
        Change model_path to match your actual .pt file name
        """
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = Path(model_path)  # This should point to your .pt file
        
        logger.info(f"Initializing LaMa with .pt model: {self.model_path}")
        logger.info(f"Using device: {self.device}")
        self.load_pt_model()

    def _setup_device(self, device):
        """Setup the computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU")
        return device

    def load_pt_model(self):
        """Load the .pt model file (REPLACE THE OLD load_model METHOD)"""
        try:
            # Check if model file exists
            if not self.model_path.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                logger.error("Please place your .pt model file in the project directory")
                self._print_pt_setup_instructions()
                return False

            # Get file size for verification
            file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"📊 Model file size: {file_size_mb:.1f} MB")
            
            if file_size_mb < 50:
                logger.warning("⚠️ Model file seems small for LaMa model")
            elif file_size_mb > 500:
                logger.warning("⚠️ Model file seems large for LaMa model")

            # Load the .pt model
            logger.info(f"📥 Loading .pt model from: {self.model_path}")
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            
            logger.info("✅ .pt LaMa model loaded successfully!")
            logger.info(f"📋 Model type: {type(self.model)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading .pt model: {e}")
            logger.error(traceback.format_exc())
            return False

    def _print_pt_setup_instructions(self):
        """Print setup instructions for .pt model"""
        instructions = f"""
        🔧 .pt Model Setup Instructions:
        
        1. Place your LaMa .pt model file in the project directory
        2. Rename it to match the model_path in the code, or update the code
        3. Current expected path: {self.model_path}
        4. Available .pt files in current directory:
        """
        
        # List available .pt files
        pt_files = list(Path(".").glob("*.pt"))
        if pt_files:
            instructions += "\n   Available .pt files:"
            for pt_file in pt_files:
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                instructions += f"\n   - {pt_file} ({size_mb:.1f} MB)"
        else:
            instructions += "\n   ❌ No .pt files found in current directory"
        
        print(instructions)

    def preprocess_image_and_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced preprocessing for .pt model"""
        logger.info(f"Input image shape: {image.shape}, mask shape: {mask.shape}")
        
        # Ensure image is in RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Advanced mask processing
        mask_processed = self._process_mask_advanced(mask)
        
        # Log mask statistics
        mask_stats = {
            'unique_values': np.unique(mask_processed),
            'non_zero_pixels': np.count_nonzero(mask_processed),
            'total_pixels': mask_processed.size,
            'percentage_masked': (np.count_nonzero(mask_processed) / mask_processed.size) * 100
        }
        logger.info(f"Processed mask stats: {mask_stats}")
        
        if mask_stats['non_zero_pixels'] == 0:
            logger.warning("⚠️ No mask detected! The mask appears to be empty.")
            return None, None
        
        # Resize mask to match image if needed
        if image.shape[:2] != mask_processed.shape[:2]:
            logger.info(f"Resizing mask from {mask_processed.shape} to {image.shape[:2]}")
            mask_processed = cv2.resize(mask_processed, (image.shape[1], image.shape[0]))
        
        # Normalize to [0, 1]
        image_norm = image.astype(np.float32) / 255.0
        mask_norm = mask_processed.astype(np.float32) / 255.0
        
        # Convert to tensors with correct dimensions [B, C, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        logger.info(f"Final tensor shapes - Image: {image_tensor.shape}, Mask: {mask_tensor.shape}")
        
        return image_tensor, mask_tensor

    def _process_mask_advanced(self, mask: np.ndarray) -> np.ndarray:
        """Advanced mask processing with multiple detection strategies"""
        strategies = [
            ("Red Channel", self._extract_red_channel_mask),
            ("Green Channel", self._extract_green_channel_mask),
            ("Blue Channel", self._extract_blue_channel_mask),
            ("Brightness", self._extract_brightness_mask),
            ("Color Difference", self._extract_color_difference_mask),
        ]
        
        best_mask = None
        best_strategy = None
        best_pixels = 0
        
        # Try each strategy and find the most reasonable one
        for strategy_name, strategy_func in strategies:
            try:
                processed_mask = strategy_func(mask)
                pixel_count = np.count_nonzero(processed_mask)
                pixel_percentage = (pixel_count / processed_mask.size) * 100
                
                logger.info(f"{strategy_name} strategy: {pixel_count} pixels ({pixel_percentage:.1f}%)")
                
                # Skip strategies that detect too much (likely noise) or too little
                if pixel_percentage > 80 or pixel_percentage < 0.1:
                    logger.info(f"  Skipping {strategy_name} - unreasonable coverage")
                    continue
                
                if pixel_count > best_pixels and pixel_percentage < 50:  # Reasonable upper limit
                    best_pixels = pixel_count
                    best_mask = processed_mask
                    best_strategy = strategy_name
                    
            except Exception as e:
                logger.warning(f"{strategy_name} strategy failed: {e}")
                continue
        
        if best_mask is None:
            logger.warning("All strategies failed, trying conservative approach")
            best_mask = self._conservative_mask_detection(mask)
            best_strategy = "Conservative"
        
        logger.info(f"Selected strategy: {best_strategy}")
        
        # Post-process the mask
        best_mask = self._post_process_mask(best_mask)
        
        return best_mask

    def _extract_red_channel_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract mask from red channel (common for drawing apps)"""
        if len(mask.shape) == 3:
            red_channel = mask[:, :, 2]  # Red channel in BGR
        else:
            red_channel = mask
        
        binary_mask = (red_channel > 50).astype(np.uint8) * 255
        return binary_mask

    def _extract_brightness_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract mask based on brightness"""
        if len(mask.shape) == 3:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            gray = mask
        
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_mask

    def _extract_color_difference_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract mask based on color difference from background"""
        if len(mask.shape) != 3:
            return np.zeros_like(mask)
        
        diff = np.sum(mask.astype(np.float32), axis=2)
        binary_mask = (diff > 50).astype(np.uint8) * 255
        return binary_mask

    def _extract_green_channel_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract mask from green channel"""
        if len(mask.shape) == 3:
            green_channel = mask[:, :, 1]  # Green channel in BGR
        else:
            green_channel = mask
        
        binary_mask = (green_channel > 50).astype(np.uint8) * 255
        return binary_mask

    def _extract_blue_channel_mask(self, mask: np.ndarray) -> np.ndarray:
        """Extract mask from blue channel"""
        if len(mask.shape) == 3:
            blue_channel = mask[:, :, 0]  # Blue channel in BGR
        else:
            blue_channel = mask
        
        binary_mask = (blue_channel > 50).astype(np.uint8) * 255
        return binary_mask

    def _conservative_mask_detection(self, mask: np.ndarray) -> np.ndarray:
        """Conservative mask detection for edge cases"""
        if len(mask.shape) == 3:
            mask_sum = np.sum(mask.astype(np.float32), axis=2)
            binary_mask = (mask_sum > 150).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 100).astype(np.uint8) * 255
        
        return binary_mask

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process mask with morphological operations"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Dilate slightly for better coverage
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        return mask

    def pad_to_modulo(self, tensor, modulo=8):
        """Pad tensor to be divisible by modulo"""
        h, w = tensor.shape[-2:]
        pad_h = (modulo - h % modulo) % modulo
        pad_w = (modulo - w % modulo) % modulo
        
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        return tensor, (pad_h, pad_w)

    def inpaint_with_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform inpainting using the .pt model (MAIN INPAINTING METHOD)"""
        if self.model is None:
            logger.warning("Model not loaded, using fallback")
            return self.advanced_fallback_inpaint(image, mask)

        try:
            logger.info("🎨 Starting .pt LaMa inpainting process...")
            
            # Preprocess inputs
            image_tensor, mask_tensor = self.preprocess_image_and_mask(image, mask)
            
            if image_tensor is None or mask_tensor is None:
                logger.error("Preprocessing failed")
                return self.advanced_fallback_inpaint(image, mask)
            
            original_shape = image_tensor.shape[-2:]
            
            # Pad tensors to be divisible by 8 (common requirement for inpainting models)
            image_padded, pad_info = self.pad_to_modulo(image_tensor, 8)
            mask_padded, _ = self.pad_to_modulo(mask_tensor, 8)
            
            logger.info(f"Padded shapes - Image: {image_padded.shape}, Mask: {mask_padded.shape}")
            
            # Run inference with the .pt model
            with torch.no_grad():
                # Try different input formats that .pt LaMa models might expect
                try:
                    # Method 1: Two separate arguments (image, mask)
                    result = self.model(image_padded, mask_padded)
                    logger.info("✅ Method 1 (two args) worked")
                except Exception as e1:
                    logger.info(f"Method 1 failed: {e1}")
                    try:
                        # Method 2: Concatenated input [image, mask] along channel dimension
                        input_tensor = torch.cat([image_padded, mask_padded], dim=1)
                        result = self.model(input_tensor)
                        logger.info("✅ Method 2 (concatenated) worked")
                    except Exception as e2:
                        logger.info(f"Method 2 failed: {e2}")
                        try:
                            # Method 3: Dictionary input
                            batch_dict = {'image': image_padded, 'mask': mask_padded}
                            result = self.model(batch_dict)
                            logger.info("✅ Method 3 (dict) worked")
                        except Exception as e3:
                            logger.error(f"All input methods failed: {e1}, {e2}, {e3}")
                            return self.advanced_fallback_inpaint(image, mask)
                
                # Handle different output formats
                if isinstance(result, dict):
                    if 'inpainted' in result:
                        inpainted = result['inpainted']
                    elif 'pred_img' in result:
                        inpainted = result['pred_img']
                    elif 'output' in result:
                        inpainted = result['output']
                    else:
                        inpainted = list(result.values())[0]
                elif isinstance(result, (list, tuple)):
                    inpainted = result[0]
                else:
                    inpainted = result
                
                logger.info(f"Inpainted tensor shape: {inpainted.shape}")
                
                # Remove padding
                pad_h, pad_w = pad_info
                if pad_h > 0 or pad_w > 0:
                    inpainted = inpainted[:, :, :original_shape[0], :original_shape[1]]
                
                # Convert to numpy
                result_np = inpainted[0].permute(1, 2, 0).cpu().numpy()
                result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
                
                # Ensure correct output size
                if result_np.shape[:2] != image.shape[:2]:
                    result_np = cv2.resize(result_np, (image.shape[1], image.shape[0]))
                
                # Convert back to BGR for OpenCV consistency
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                
                logger.info("✅ .pt LaMa inpainting completed successfully!")
                return result_bgr
                
        except Exception as e:
            logger.error(f"❌ .pt LaMa inpainting failed: {e}")
            logger.error(traceback.format_exc())
            return self.advanced_fallback_inpaint(image, mask)

    def advanced_fallback_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advanced fallback inpainting with multiple OpenCV algorithms"""
        logger.info("🔄 Using advanced fallback inpainting...")
        
        # Process mask
        mask_processed = self._process_mask_advanced(mask)
        
        if np.count_nonzero(mask_processed) == 0:
            logger.warning("No valid mask found, returning original image")
            return image.copy()
        
        # Try different inpainting methods
        methods = [
            (cv2.INPAINT_TELEA, "Telea"),
            (cv2.INPAINT_NS, "Navier-Stokes")
        ]
        
        best_result = None
        best_score = -1
        
        for method, name in methods:
            try:
                result = cv2.inpaint(image, mask_processed, inpaintRadius=7, flags=method)
                
                # Simple quality score based on variance in inpainted region
                masked_region = result[mask_processed > 0]
                if len(masked_region) > 0:
                    score = np.var(masked_region)
                    logger.info(f"{name} method score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"{name} method failed: {e}")
                continue
        
        if best_result is not None:
            logger.info("✅ Fallback inpainting completed")
            return best_result
        else:
            logger.warning("All fallback methods failed, returning original")
            return image.copy()

# ===============================
# CHANGE THIS LINE TO MATCH YOUR .pt FILE NAME!
# ===============================
try:
    inpainter = AdvancedLaMaInpainter(model_path="big-lama.pt", device="auto")  # <-- CHANGE THIS NAME
    model_status = "✅ Ready" if inpainter.model is not None else "⚠️ Fallback Mode"
except Exception as e:
    logger.error(f"Failed to initialize inpainter: {e}")
    inpainter = None
    model_status = "❌ Failed"

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:  # Add encoding='utf-8'
            return f.read()
    except FileNotFoundError:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Object Remover API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .ready {{ background-color: #d4edda; color: #155724; }}
                .fallback {{ background-color: #fff3cd; color: #856404; }}
                .failed {{ background-color: #f8d7da; color: #721c24; }}
            </style>
        </head>
        <body>
            <h1>🎨 Advanced Object Remover API (.pt model)</h1>
            <div class="status {model_status.split()[0].lower().replace('✅', 'ready').replace('⚠️', 'fallback').replace('❌', 'failed')}">
                <strong>Status:</strong> {model_status}
            </div>
            <p>API is running! Device: <strong>{inpainter.device if inpainter else 'Unknown'}</strong></p>
            <h3>Available endpoints:</h3>
            <ul>
                <li><strong>POST /api/remove</strong> - Remove objects from images</li>
                <li><strong>GET /api/health</strong> - Check API health</li>
                <li><strong>POST /api/test</strong> - Test with sample data</li>
            </ul>
        </body>
        </html>
        """

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    if inpainter is None:
        return jsonify({
            "status": "error",
            "message": "Inpainter not initialized"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "model_loaded": inpainter.model is not None,
        "device": inpainter.device,
        "cuda_available": torch.cuda.is_available(),
        "model_type": ".pt model",
        "model_path": str(inpainter.model_path),
        "version": "Enhanced LaMa v2.0"
    })

@app.route('/api/remove', methods=['POST'])
def remove_object():
    """Enhanced object removal endpoint"""
    if inpainter is None:
        return jsonify({"error": "Inpainter not initialized"}), 500
    
    try:
        logger.info("🔥 Received object removal request...")
        
        # Validate request
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({"error": "Both image and mask files are required"}), 400
        
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        if not image_file.filename or not mask_file.filename:
            return jsonify({"error": "Invalid file names"}), 400
        
        logger.info(f"📁 Processing files: {image_file.filename}, {mask_file.filename}")
        
        # Read and decode images
        image_bytes = image_file.read()
        mask_bytes = mask_file.read()
        
        logger.info(f"📊 File sizes - Image: {len(image_bytes)} bytes, Mask: {len(mask_bytes)} bytes")
        
        nparr_img = np.frombuffer(image_bytes, np.uint8)
        nparr_mask = np.frombuffer(mask_bytes, np.uint8)
        
        image = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
        mask = cv2.imdecode(nparr_mask, cv2.IMREAD_COLOR)
        
        if image is None or mask is None:
            return jsonify({"error": "Could not decode images"}), 400
        
        logger.info(f"🖼️ Decoded shapes - Image: {image.shape}, Mask: {mask.shape}")
        
        # Perform enhanced inpainting
        result = inpainter.inpaint_with_lama(image, mask)
        
        # Encode and return result
        success, encoded_img = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            return jsonify({"error": "Could not encode result image"}), 500
        
        logger.info("🎉 Successfully processed and returning result!")
        
        return send_file(
            io.BytesIO(encoded_img.tobytes()),
            mimetype='image/png',
            as_attachment=False,
            download_name='inpainted_result.png'
        )
        
    except Exception as e:
        logger.error(f"💥 Error processing request: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/test', methods=['POST'])
def test_inpainting():
    """Enhanced test endpoint"""
    if inpainter is None:
        return jsonify({"error": "Inpainter not initialized"}), 500
    
    try:
        logger.info("🧪 Running inpainting test...")
        
        # Create enhanced test image
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        # Add a more complex background
        cv2.rectangle(test_image, (50, 50), (550, 350), (200, 220, 255), -1)
        cv2.circle(test_image, (150, 150), 30, (255, 200, 200), -1)
        cv2.circle(test_image, (450, 250), 40, (200, 255, 200), -1)
        
        # Add watermark/object to remove
        cv2.rectangle(test_image, (200, 150), (400, 250), (0, 0, 0), -1)
        cv2.putText(test_image, "REMOVE ME", (210, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create mask for the object
        test_mask = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(test_mask, (200, 150), (400, 250), (0, 0, 255), -1)  # Red mask
        
        # Perform inpainting
        result = inpainter.inpaint_with_lama(test_image, test_mask)
        
        # Encode result
        success, encoded_img = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            return jsonify({"error": "Could not encode test image"}), 500
        
        logger.info("✅ Test completed successfully!")
        
        return send_file(
            io.BytesIO(encoded_img.tobytes()),
            mimetype='image/png',
            as_attachment=False,
            download_name='test_result.png'
        )
        
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Advanced Object Remover API...")
    print(f"📱 Device: {inpainter.device if inpainter else 'Unknown'}")
    print(f"🤖 Model Status: {model_status}")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)