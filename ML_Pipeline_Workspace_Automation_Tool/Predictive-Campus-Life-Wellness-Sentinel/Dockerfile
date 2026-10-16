# syntax=docker/dockerfile:1
FROM python:3.12

# Setting up the working directory for the REST app
WORKDIR /app

#Copying the Flask-RESTfull service to alpine's working directory
COPY ./* /app/

# RUN the requirements to install the python dependies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose application port
EXPOSE 9104

# Production-ready command
CMD ["gunicorn", "-c", "gConfig.py", "rApi:app"]
