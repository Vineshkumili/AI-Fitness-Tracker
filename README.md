# 🏋️ AI Fitness Tracker

A **computer vision–based fitness tracking system** built using **OpenCV** and **MediaPipe Pose**.  
This project detects human body landmarks in real-time through a webcam and **automatically counts repetitions** of different exercises or tracks **duration** (for planks).  

The tracker provides **on-screen feedback** for accuracy and progress, making home workouts more engaging, measurable, and interactive.  

---

## ✨ Features

- **Real-time pose detection** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose).  
- Supports **6 common exercises**:
  1. Push-ups (reps)
  2. Sit-ups (reps)
  3. Squats (reps)
  4. Jumping Jacks (reps)
  5. Lunges (reps)
  6. Plank (time duration)
- **Automatic rep counting** & plank timer.
- **On-screen feedback** with repetition counts, form corrections, or timer updates.
- **Modular exercise detection**: each exercise has its own detection script.
- **Keyboard input** to choose the exercise mode.
- Easily extendable to add more exercises.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** → Video capture, processing, and visualization.
- **MediaPipe Pose** → Real-time body landmark detection.
- **NumPy** → Mathematical operations (angles, distance, threshold calculations).
- **PyAutoGUI** (optional) → Extend for automation or GUI control.

---

## 📂 Project Structure

```
AI-Fitness-Tracker/
│── exercise_tracker.py   # Main runner script
│── utils/                # Utility functions (angle, drawing, etc.)
│   ├── pushup.py         # Push-up detection logic
│   ├── situp.py          # Sit-up detection logic
│   ├── squat.py          # Squat detection logic
│   ├── jumpingjack.py    # Jumping jack detection logic
│   ├── lunge.py          # Lunge detection logic
│   ├── plank.py          # Plank detection logic (with timer)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Vineshkumili/AI-Fitness-Tracker.git
cd AI-Fitness-Tracker
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
(or install manually)  
```bash
pip install opencv-python mediapipe numpy
```

### 3️⃣ Run the Tracker
```bash
python exercise_tracker.py
```

### 4️⃣ Select Exercise
Choose an exercise by pressing the corresponding key in the terminal:  
- `1` → Push-ups  
- `2` → Sit-ups  
- `3` → Squats  
- `4` → Jumping Jacks  
- `5` → Lunges  
- `6` → Plank  

---

## 🎥 Demo (Coming Soon)

We will add a short video showcasing real-time detection and counting.

---

## 🧩 Future Improvements

- 🔢 **Calorie estimation** based on body weight & activity.  
- 🗣️ **Voice feedback** (e.g., “Good squat!”, “Keep your back straight!”).  
- 🖥️ **GUI-based mode selection** instead of CLI.  
- 📊 **Logging & analytics dashboard** for progress tracking.  
- 📱 **Mobile/IoT integration** for smart gym assistants.  

---

## 🏃 Use Cases

- Personal fitness tracking at home.  
- Gamifying workouts with real-time feedback.  
- Data-driven progress monitoring for fitness enthusiasts.  
- Integration with smart mirrors or gym equipment.  

---

## 👨‍💻 Contributing

Contributions are welcome!  
To contribute:  
1. Fork this repository.  
2. Create a new branch (`feature-branch`).  
3. Commit your changes.  
4. Submit a Pull Request.  

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/)  
- [MediaPipe](https://developers.google.com/mediapipe)  
- Inspiration from fitness & health AI applications  

---

### 💡 Author
Developed by **Vinesh Kumili** ✨  
For questions or suggestions, feel free to connect on GitHub.  
