{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "XozrdvMmZZNO"
      },
      "outputs": [],
      "source": [
        "import cv2\n",
        "import numpy as np\n",
        "import dlib\n",
        "from imutils import face_utils\n",
        "from google.colab.patches import cv2_imshow"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the threshold for detecting drowsiness\n",
        "DROWSINESS_THRESHOLD = 10\n",
        "DROWSINESS_FRAMES = 5"
      ],
      "metadata": {
        "id": "3hSiNufCaGwI"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to calculate distance between two points\n",
        "def calculate_distance(a, b):\n",
        "    x1, y1 = a\n",
        "    x2, y2 = b\n",
        "    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5"
      ],
      "metadata": {
        "id": "8rr94_-2adTu"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to detect drowsiness\n",
        "def detect_drowsiness(dlist):\n",
        "    return sum(dlist) >= DROWSINESS_FRAMES"
      ],
      "metadata": {
        "id": "MLKLqvWGamFp"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Define the function to capture a photo\n",
        "def take_photo(quality=0.8):\n",
        "    from IPython.display import display, Javascript\n",
        "    from google.colab.output import eval_js\n",
        "    from base64 import b64decode\n",
        "\n",
        "    js = Javascript('''\n",
        "        async function takePhoto(quality) {\n",
        "            const div = document.createElement('div');\n",
        "            const capture = document.createElement('button');\n",
        "            capture.textContent = 'Capture';\n",
        "            div.appendChild(capture);\n",
        "\n",
        "            const video = document.createElement('video');\n",
        "            video.style.display = 'block';\n",
        "            const stream = await navigator.mediaDevices.getUserMedia({video: true});\n",
        "\n",
        "            document.body.appendChild(div);\n",
        "            div.appendChild(video);\n",
        "            video.srcObject = stream;\n",
        "            await video.play();\n",
        "\n",
        "            // Resize the output to fit the video element.\n",
        "            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);\n",
        "\n",
        "            // Wait for Capture to be clicked.\n",
        "            await new Promise((resolve) => capture.onclick = resolve);\n",
        "\n",
        "            const canvas = document.createElement('canvas');\n",
        "            canvas.width = video.videoWidth;\n",
        "            canvas.height = video.videoHeight;\n",
        "            canvas.getContext('2d').drawImage(video, 0, 0);\n",
        "            stream.getVideoTracks()[0].stop();\n",
        "            div.remove();\n",
        "            return canvas.toDataURL('image/jpeg', quality);\n",
        "        }\n",
        "    ''')\n",
        "    display(js)\n",
        "    data = eval_js('takePhoto({})'.format(quality))\n",
        "    binary = b64decode(data.split(',')[1])\n",
        "    return binary"
      ],
      "metadata": {
        "id": "GNCPGzRs1SCs"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import urllib.request\n",
        "\n",
        "# URL of the shape predictor file\n",
        "shape_predictor_url = \"http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\"\n",
        "# Local file path where the shape predictor file will be saved\n",
        "shape_predictor_file = \"shape_predictor_68_face_landmarks.dat.bz2\"\n",
        "\n",
        "# Download the shape predictor file\n",
        "print(\"Downloading shape predictor file...\")\n",
        "urllib.request.urlretrieve(shape_predictor_url, shape_predictor_file)\n",
        "print(\"Shape predictor file downloaded successfully!\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5SEVgflTnQJ7",
        "outputId": "4a130f4e-e84b-45f1-f5a5-aaf8f7f18679"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading shape predictor file...\n",
            "Shape predictor file downloaded successfully!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the main function\n",
        "def main():\n",
        "    # Initialize the face detector and shape predictor\n",
        "    detector = dlib.get_frontal_face_detector()\n",
        "    predictor = dlib.shape_predictor(\"/content/shape_predictor_68_face_landmarks.dat\")\n",
        "\n",
        "    dlist = []\n",
        "\n",
        "    while True:\n",
        "        frame_data = take_photo()  # Call the take_photo function to capture a frame\n",
        "        nparr = np.frombuffer(frame_data, np.uint8)\n",
        "        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)\n",
        "\n",
        "        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
        "        rects = detector(gray, 0)\n",
        "\n",
        "        for (i, rect) in enumerate(rects):\n",
        "            shape = predictor(gray, rect)\n",
        "            shape = face_utils.shape_to_np(shape)\n",
        "\n",
        "            le_38 = shape[37]\n",
        "            le_39 = shape[38]\n",
        "            le_41 = shape[40]\n",
        "            le_42 = shape[41]\n",
        "\n",
        "            re_44 = shape[43]\n",
        "            re_45 = shape[44]\n",
        "            re_47 = shape[46]\n",
        "            re_48 = shape[47]\n",
        "\n",
        "            # Calculate the average distance between facial landmarks\n",
        "            avg_distance = (calculate_distance(le_38, le_42) + calculate_distance(le_39, le_41) +\n",
        "                            calculate_distance(re_44, re_48) + calculate_distance(re_45, re_47)) / 4\n",
        "\n",
        "            # Append the average distance to the list\n",
        "            dlist.append(avg_distance)\n",
        "\n",
        "            # Check if the length of dlist exceeds the threshold for consecutive frames\n",
        "            if len(dlist) > DROWSINESS_FRAMES:\n",
        "                dlist.pop(0)\n",
        "\n",
        "            # Check if drowsiness is detected based on the average distance and threshold\n",
        "            if detect_drowsiness(dlist):\n",
        "                cv2.putText(frame, \"Drowsiness Detected\", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\n",
        "            else:\n",
        "                cv2.putText(frame, \"No Drowsiness\", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)\n",
        "        print(\"Output\")\n",
        "        cv2_imshow(frame)\n",
        "\n",
        "        if cv2.waitKey(1) & 0xFF == 27:\n",
        "            break\n",
        "\n",
        "    cv2.destroyAllWindows()"
      ],
      "metadata": {
        "id": "v7e2f6yGatJf"
      },
      "execution_count": 36,
      "outputs": []
    }
  ]
}