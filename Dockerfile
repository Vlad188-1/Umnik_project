# установка базового образа (host OS)
FROM ubuntu:22.04
RUN apt update && apt -y install sudo

## создание пользователя vlad
RUN adduser --disabled-password --gecos "" vlad && \
    usermod -aG sudo vlad && \
    echo "%sudo  ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/nopasswd

WORKDIR /home/vlad
USER vlad
CMD bash

## установка пакетов для linux и корректной работы GUI
RUN sudo apt update && sudo apt install -y libgl1-mesa-glx libpci3 libasound2 \
    '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev  \
    libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libglib2.0-0 libfontconfig1 libdbus-1-3 wget

WORKDIR /home/vlad
COPY ./ /home/vlad/app
RUN sudo chown -R vlad /home/vlad/app

## скачивание и установка miniconda, устанвика pyqt и других библиотек
RUN cd /home/vlad && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/vlad/miniconda3 \
    && /bin/bash -c "source ./miniconda3/etc/profile.d/conda.sh && conda activate base && conda init bash && \
                    conda install -c anaconda pyqt && \
                    conda install pip && pip install -r /home/vlad/app/requirements.txt" \
## команда, выполняемая при запуске контейнера
CMD [ "python3", "./app/app.py" ]
