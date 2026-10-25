# Face Attendance

Take attendance with face recognition, right in your browser.

- Open the app, press **Start camera**, add a person (40 photos), press **Train model**.
- Stand in front of the camera and you are checked in with the time.
- **Download CSV** saves the list.
- It can be installed on a phone or laptop like an app.

Built with Python, OpenCV (Haar cascade + LBPH) and Flask. The camera runs in the visitor's
browser, and small pictures are sent to the server for detection and recognition.

## Run it on your computer

Use the standard Python from python.org (3.10 to 3.13 is safest). On Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

On Mac or Linux: `python3 -m venv venv`, then `venv/bin/pip install -r requirements.txt` and `venv/bin/python app.py`.

Open http://127.0.0.1:7860 for the website and http://127.0.0.1:7860/app for the app.
OpenCV 5 is not supported yet, so `requirements.txt` keeps OpenCV below version 5.

## Deploy it on Render (free)

1. Put these files in a GitHub repository (the `Dockerfile` must be at the top level).
2. On render.com choose **New + → Web Service**, connect the repository, set **Language** to **Docker**
   and **Instance type** to **Free**, then deploy.
3. Open `https://YOUR-NAME.onrender.com/app`.

The free plan sleeps after 15 minutes without visitors and has little CPU, so the first visit is slow
and recognition can lag.

## Notes

- People, photos and the trained model live only in the server's memory. They are cleared after
  30 minutes without use and whenever the app restarts.
- Each visitor has a separate session (up to 5 people each, 20 visitors at once).
- Face recognition here uses a simple classic method (LBPH). It works best in similar lighting to
  when someone was added, and a printed photo can fool it.
