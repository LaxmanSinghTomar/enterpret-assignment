# Using Python3.9.13-slim-buster as Base Docker image
FROM python:3.9.16-slim-buster

# Defining Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/laxman

# Create Working-Directory
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# System Library Installation
RUN apt update && \
apt install gcc g++ -y --no-install-recommends

# Virtual Environment Creation & requirements.txt Installation
RUN python3 -m venv /venv

# Python Environment Path
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH

# Create the app user called emplay
RUN adduser --disabled-password --gecos '' laxman

# Install python packages
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
pip install --no-cache-dir -r requirements.txt

# Copy Repo & Dependencies
COPY . /$APP_HOME

# Remove unwanted packages
RUN rm -rf /var/lib/apt/lists/* && \
apt-get autoremove -y && apt-get autoclean -y && \
rm -rf ~/.cache/pip

# Change the Folder Permissions to emplay user & Log Folder Creation
RUN chown -R laxman:laxman $APP_HOME && \
chmod -R 755 $APP_HOME

#Change the user to start the app
USER laxman

EXPOSE 9200

ENTRYPOINT ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "9200"]