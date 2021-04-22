## Installation 
* pip install -r requirements


## Dependencies 
1. Clone [OpenDBM](https://github.com/AiCure/open_dbm)
2. Install requirements for OpenDBM
3. Clone binaries from OpenFace [here](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip) and specify path to FaceLandmarkVid.exe in OpenDBM's process_data.py file
4. Download protoFile = pose_deploy_linevec.prototxt for body class in main.py 
6. Download caffemodel pose_iter_160000.caffemodel for body class in main.py


## Usage
* Specify OpenDBM's process_data.py file path in main.py
* Specify protoFile path in main.py
* Specify caffemodel path in main.py
* run in command prompt "python main.py"

## Next steps 

* Package application in an exe file using [pyinstaller](https://www.pyinstaller.org/) or [fbs](https://build-system.fman.io/)
* If your build goes well, feel free to share! If it doesn't, contact me @ [chags1313@gmail.com](chags1313@gmail.com)

The goal is to have a working exe for all to use in the future. This will take some time, since we still need to get all functions working properly and clean the code and UI. 
