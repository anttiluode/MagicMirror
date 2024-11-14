# Magic Mirror

Magic Mirror is an AI-enhanced edge detection application that uses Stable Diffusion 2.1 to apply style transfer to webcam images. This project allows users to experiment with AI image generation in real-time, using either edge-detected versions or the original video feed.

## Features
- **Live Video Feed**: Uses your webcam to capture real-time video.
- **Edge Detection**: Applies CV2 edge detection and allows manipulation through different filters.
- **AI Style Transfer**: Utilizes Stable Diffusion 2.1 to generate AI-styled images based on the webcam feed.
- **Customizable Parameters**: Adjust the strength of the AI style transfer, guidance scale, and various other visual effects.

## Requirements
- Python 3.8+
- Stable Diffusion 2.1 (`diffusers` library)
- OpenCV (`cv2`)
- Torch (`torch`)
- Tkinter (for GUI)
- Pillow (`PIL`)

## Installation

**Clone the repository**:

   git clone https://github.com/anttiluode/MagicMirror.git
   cd magic-mirror
   
**Install the required dependencies**:

   pip install -r requirements.txt
   
**Note**: Make sure to have a compatible CUDA version installed if you plan to run Stable Diffusion on a GPU.

## Running Magic Mirror

   python app.py

2. **Open Camera**: Select a camera and click "Open Camera".

3. **Apply Edge Detection or Effects**: Use the sliders and toggles to adjust brightness, contrast, blur, etc.

4. **Start AI Rendering**: Click "Start AI Rendering" to generate AI-styled images in real-time based on either the edge-detected or original webcam feed.

## Dependencies

- `diffusers`: To load the Stable Diffusion 2.1 model.
- `opencv-python`: For webcam input and edge detection.
- `torch`: To run the Stable Diffusion model.
- `Pillow`: To manipulate and display images in the Tkinter GUI.

## Notes

- **Stable Diffusion Model**: This project uses Stable Diffusion 2.1, which is capable of producing high-quality AI-generated images. Make sure you have enough system resources if running on CPU. It can be easily changed by asking chatgpt etc to do it.
- 
- **Hardware Requirements**: It is recommended to run this on a machine with a CUDA-compatible GPU to achieve reasonable performance. I am using 3060ti and it does a image perhaps every 10 seconds. 

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.

## Acknowledgements

- **Stable Diffusion**: Thanks to the Stability AI team for creating Stable Diffusion 2.1.

- **Diffusers Library**: Hugging Face's `diffusers` library makes it easy to work with diffusion models.
