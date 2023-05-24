# установка базового образа (host OS)
FROM python:3.10
RUN apt update && apt install -y libgl1-mesa-glx libpci3 libasound2

WORKDIR /home/app
COPY ./ /home/app

RUN cd /home && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && chmod +x Miniconda3-latest-Linux-aarch64.sh \
    && ./Miniconda3-latest-Linux-aarch64.sh -b -p /home/miniconda3 \
    && /bin/bash -c "source ./miniconda3/etc/profile.d/conda.sh && conda activate base && conda init bash && \
                    conda install -c anaconda pyqt && conda install pip && conda install -c conda-forge pyqtwebengine && \
                    conda install pip && pip install -r /home/app/requirements.txt" \
#WORKDIR /home/app
#COPY ./ /home/app
#RUN pip install -r /home/app/requirements.txt
# установка рабочей директории в контейнере
#WORKDIR /app
## копирование файла зависимостей в рабочую директорию
#COPY requirements.txt .
## установка зависимостей
#RUN pip install -r requirements.txt
## копирование содержимого локальной директории src в рабочую директорию
#COPY ./ .
## команда, выполняемая при запуске контейнера
#CMD [ "python3", "./app.py" ]