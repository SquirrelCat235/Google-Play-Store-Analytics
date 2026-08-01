import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def verify_gpu():
    print("==================================================")
    print(" GPU Hardware Acceleration Verification Test      ")
    print("==================================================")
    
    version = torch.__version__
    cuda_ver = torch.version.cuda
    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else "None"
    
    print(f"  - PyTorch Version : {version}")
    print(f"  - CUDA Version    : {cuda_ver}")
    print(f"  - CUDA Available  : {available}")
    print(f"  - GPU Model       : {device_name}")
    
    if not available:
        raise RuntimeError("CUDA is not available!")
        
    device = torch.device("cuda:0")
    print(f"\n[Testing GPU Tensor Allocation & Autocast FP16]")
    x = torch.randn(16, 64, 768, device=device)
    print(f"  - Allocated tensor on {x.device}: shape {x.shape}")
    
    print("\n[Loading DistilBERT to RTX 2050 GPU]")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(device)
    print(f"  - Model moved to GPU successfully ({next(model.parameters()).device})")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(["This Google Play app is amazing and smooth!"], return_tensors="pt").to(device)
    
    with torch.amp.autocast("cuda"):
        outputs = model(**inputs)
        
    print(f"  - FP16 Autocast Forward Pass Output Shape: {outputs.logits.shape}")
    print(f"  - Predicted Logits: {outputs.logits.detach().cpu().numpy()}")
    print("\n✅ GPU Execution Verification PASSED Successfully!")

if __name__ == "__main__":
    verify_gpu()
