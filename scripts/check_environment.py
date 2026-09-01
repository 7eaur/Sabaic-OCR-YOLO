#!/usr/bin/env python3
import json,platform
import numpy,PIL,torch
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"numpy":numpy.__version__,"pillow":PIL.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
