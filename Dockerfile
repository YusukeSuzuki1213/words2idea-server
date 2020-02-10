FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
        cmake \
        git \
        curl \
		emacs \
		wget \
		llvm \
		libncurses5-dev \
		libncursesw5-dev \
		xz-utils \
		tk-dev \
	 	libffi-dev \ 
	 	liblzma-dev \
	 	python-openssl \
	 	make \
	 	libssl-dev \
	 	zlib1g-dev \
	 	libbz2-dev \
	 	libreadline-dev \
	 	libsqlite3-dev \
	 	libffi-dev && \
     	rm -rf /var/lib/apt/lists/*

RUN apt-get update

# install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN . ~/.bashrc

# 下記のエラーが出力されるため、これ以降はコンテナが立ち上がったらインストする
# "/bin/bash: pyenv: command not found"
#RUN pyenv install 3.7.4
#RUN pyenv install 2.7.16

# default dir
WORKDIR /mnt
#RUN pyenv global 3.7.4
#RUN pyenv local  3.7.4

#RUN pip3 pipenv install
#RUN pipenv run python3 -m spacy download en_vectors_web_lg
