# AI-HANDSIGN 🤚🌐

## Description 📋

AI-HANDSIGN is a project that utilizes Mediapipe technology and deep learning models to translate the American Sign Language (ASL) alphabet in real-time. It enables deaf and mute individuals to communicate more easily with those who do not know sign language.

## How It Works? 🚀


1. **Installation Requirements:** Run `pip install -r requirements.txt` to install the necessary dependencies.

2. **Model Training:** Ensure you have a folder with images for model training. Additionally, there's a pre-trained model named "best_model.pth" that you can use to test how it works.

3. **Running the Application:** Open the `main.py` file and select the desired camera. Then, run the program.

4. **Real-time Translation:** The application displays the ASL alphabet in real-time as it detects hand gestures.

5. **Model Training:** To train the model, follow these steps:
   - Create a destination folder where you want to place the images.
   - Inside the destination folder, create individual folders for each letter from A to Z (excluding the letter 'Ñ').
   - Additionally, create three extra categories named 'del,' 'space,' and 'nothing.'
   - Organize your image dataset like this:
     ```
     - destination-folder
       -- A
         --- A1.jpg
         --- A2.jpg
         --- A3.jpg
       -- B
         --- B1.jpg
         --- B2.jpg
         --- B3.jpg
       -- C
       -- del
       -- space
     ```
   - Place the images of each sign in their respective folders.
   - You can use your own dataset or find existing ASL alphabet datasets online.

6. **Additional Controls:**
   - Press the **'F' key** to flip the camera horizontally.
   - Press the **'ESC' key** to exit the program.

## Demo 👓
   🎬 [Watch the demo video here](https://drive.google.com/file/d/1u0zD73JhqAwAIc18s_nFqQMmDlpxvrH1/view?usp=sharing)  
   📄 [View the Word document here](https://docs.google.com/document/d/122KXRp3gsH-HzVKIgH_GAhoSJa6kRkkL/edit?usp=sharing&ouid=110283832961834477971&rtpof=true&sd=true) 
