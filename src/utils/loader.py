import torch
import os.path

class loader:
    def __init__(self, path):
        self.model = None
        self.path = path
        self.isCudaAvailable = torch.cuda.is_available()
        try:
            if not os.path.exists(path):
                raise FileNotFoundError("File not found")
            if not os.path.isfile(path):
                raise FileNotFoundError("Not a file")
            
            if ( self.isCudaAvailable ):
                print("Using GPU")
                self.model = torch.load(path)
            else:
                print("Using CPU")
                self.model = torch.load(path, map_location=torch.device('cpu'))
        except Exception as exc:
            raise InterruptedError(f"Error loading model ({exc})") from exc