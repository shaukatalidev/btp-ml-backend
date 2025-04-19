# 📄 ML Backend Setup Guide

This document provides step-by-step instructions for setting up and running the ML backend in both **Development** and **Production (local with ngrok)** environments.

## ⚙️ Development Setup

1. Clone the repository and navigate to the project directory.
2. Ensure you have **Python 3.8+** installed on your system.
3. Set up a virtual environment and activate it:

- **Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Linux/MacOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

4. Install the required dependencies using `pip install -r requirements.txt`.
5. Start the development server using Uvicorn: `uvicorn app:app --reload`.
6. Access the API documentation at `http://localhost:8000/docs`.

## 🚀 Production Setup (Local Machine with ngrok)

### Requirements

- **Python 3.8+**: Ensure Python is installed on your system.
- **ngrok authtoken**: Obtain your ngrok authtoken from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).
- **ngrok domain**: Configure a custom ngrok domain if required.

### Steps

1. Clone the repository (if not already cloned).
2. Configure the ngrok domain in the `docker-compose.yml` file. Specifically, update the `command` field with your ngrok domain as follows:
   ```yaml
   command: http --domain=sweet-frequently-cardinal.ngrok-free.app ml:9000
   ```
3. Run the application using Docker Compose with your ngrok authtoken: `NGROK_AUTHTOKEN={your-ngrok-authtoken} docker-compose up --build`.
4. Access the production API documentation at `https://your-ngrok-domain/docs`.

## 📝 Notes

- Always activate the virtual environment when working in development mode.
- Replace placeholder values (e.g., ngrok domain, authtoken) carefully to avoid issues.
- Keep sensitive information like `NGROK_AUTHTOKEN` secure.
- For production-grade deployments, consider using cloud services or Docker hosting solutions.

## ✅ Quick Summary

|        Environment         |               How to Run               |            Access URL            |
| :------------------------: | :------------------------------------: | :------------------------------: |
|        Development         |       `uvicorn app:app --reload`       |   `http://localhost:8000/docs`   |
| Production (local + ngrok) | `docker-compose up --build` with ngrok | `https://your-ngrok-domain/docs` |
