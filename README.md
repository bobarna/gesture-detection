# Gesture Detection
Team Project. Computer Vision Spring 2024

The project website is hosted from the `docs` folder at https://github.gatech.edu/pages/bborcsok3/gesture-detection/.

## Set up virtual environment

TLDR: `pip install -r requirements.txt`

1. Create a virtual environment with `python3 -m venv venv`. 

2. To activate this environment:
```
# MacOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Once activated, `pip` will refer to packages installed in the `venv` folder
(e.g. `venv/lib/python[your-version]/site-packages`).

3. Install required packages with given version numbers with
```
pip install -r requirements.txt
```

4. (Optional) To update the requirements, you can freeze the package numbers
   with

```
pip freeze > requirements.txt
```

5. You can deactivate the virtual environment using `deactivate`.

- Mediapipe: https://developers.google.com/mediapipe/solutions/setup_python
    - Python: version 3.8-3.11

Downgrade virtual environment to e.g. python3.11:

```
$ python3.11 -m venv --upgrade venv
$ cd venv/bin
$ ln -sf python3.11 python
$ ln -sf python3.11 python3
$ rm {python,pip}3.11
$ cd -
```
(See: https://stackoverflow.com/questions/71831415/downgrade-python-version-in-virtual-environment )

## Used resources
- OpenCV Documentation's "Getting Started with Videos": https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
- MediaPipe official demo: https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
