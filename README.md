# 🏀 Hoops Name Generator · AI Basketball Player Name Generator

A deep learning-powered web application that generates unique, realistic, and creative basketball player names using PyTorch and FastAPI. The project features an interactive, basketball court-themed UI with real-time temperature (creativity) control and a personal favorites roster.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HTML5/CSS3](https://img.shields.io/badge/UI-HTML5%20%2F%20CSS3-E34F26?style=for-the-badge&logo=html5&logoColor=white)

---

## 📌 Features

* **Character-Level Deep Learning Model:** Generates brand-new player names based on patterns learned from real-world basketball datasets.
* **Dynamic Temperature Control:** Real-time creativity adjustment via a custom basketball-themed slider (0.20 to 1.20).
* **Favorites Roster:** Save any generated name to a persistent personal roster with a single tap on the star icon, browse your saved names in a dedicated roster view, and remove them individually whenever you like.
* **FastAPI Backend:** Lightweight, high-performance REST API handling instant model inference.
* **Basketball Court-Themed UI:** Wood-grain background, chalkboard-style typography (Anton, Oswald, JetBrains Mono), and leather-accented components for an immersive court aesthetic.
* **One-Click Reset:** Instantly reset inference temperature back to the optimized default (0.70).

---

## 🛠️ Tech Stack

* **Machine Learning / Deep Learning:** PyTorch
* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML5, Modern CSS3 (custom properties, responsive design), JavaScript (Fetch API)
* **Client-Side Storage:** Browser `localStorage` for persisting the favorites roster
* **Typography:** Google Fonts — Anton, Oswald, JetBrains Mono
* **Environment:** Python 3.9+

---

## 📁 Project Structure

```text
hoops-name-generator/
│
├── data/                  # Dataset files for training
├── models/                # Trained PyTorch model weights (.pt / .pth)
├── src/                   # Core neural network architectures & generation pipeline
├── app.py                 # FastAPI application & API endpoints
├── main.py                # Application entry point
├── index.html             # Interactive frontend user interface
├── train_and_save.py      # Model training & serialization script
├── requirements.txt       # Python dependencies
├── .gitignore              # Git ignore rules (venv, __pycache__, etc.)
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/egemenilerigelen/hoops-name-generator.git
cd hoops-name-generator
```

### 2. Create & Activate Virtual Environment

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Launch the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

---

## 🎯 How Temperature (Creativity) Works

The temperature parameter scaling dictates the randomness of character probability distributions during sampling:

| Range | Behavior |
|---|---|
| **Low (0.20 – 0.50)** | Predictable, conventional, and highly realistic names. |
| **Balanced (0.70 — Default)** | Optimal balance between realism and novelty. |
| **High (0.90 – 1.20)** | High variance, wild, and creative combinations. |

---

## ⭐ Favorites Roster

Every generated name can be saved to your personal roster:

* Tap the **star icon** on a generated name to add or remove it from your favorites.
* Tap **ROSTER** in the header to switch to your saved list, and **← GENERATE** to return to the generator.
* Each roster entry can be removed individually with its **✕** button.
* The roster is stored locally in the browser, so it persists between visits on the same device.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/egemenilerigelen/hoops-name-generator/issues) if you want to contribute.
