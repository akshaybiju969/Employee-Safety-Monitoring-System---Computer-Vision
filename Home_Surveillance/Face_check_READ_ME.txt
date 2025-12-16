Face Mask Detection System
This code snippet detects whether a person’s face is masked or not using a combination of Mediapipe Pose landmarks and OpenCV Haar cascades. It analyzes the position and visibility of facial features (eyes and mouth) to estimate if a mask is being worn. When suspicious behavior is detected, it can trigger an audio warning.

Key Features

Detects frontal and back-facing individuals using pose landmarks.

Detects eyes and mouth using Haar cascades.

Determines mask-wearing status based on the visibility of eyes and mouth.

Displays “Probability = 50%” for suspected unmasked individuals.

Plays a warning sound when suspicious behavior persists.