FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

# 学外でbuildするならば，以下のENVをコメントアウト
#ENV http_proxy="http://http-p.srv.cc.suzuka-ct.ac.jp:8080/"
#ENV https_proxy="http://http-p.srv.cc.suzuka-ct.ac.jp:8080/"
#ENV no_proxy=127.0.0.1,localhost

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y python2.7-dev python3.6-dev pylint
#RUN apt install -y python-dev python-pip python-setuptools \ 
RUN apt install -y python3-pip python3-setuptools python-pip
RUN apt install -y curl
RUN apt install -y nano
RUN apt install -y git
RUN apt install -y golang-go
RUN apt install -y graphviz

RUN apt clean -y
RUN apt autoremove -y
RUN apt update -y
RUN apt upgrade -y
RUN apt autoremove -y
RUN apt autoclean -y

RUN pip install pydot
RUN pip3 install tensorflow-gpu keras
RUN pip3 install pydotplus && pip3 install graphviz
RUN pip3 install gym && pip3 install matplotlib && pip3 install && pip3 install retry
RUN cd home && git clone https://github.com/TakYusaku/sotsuken_nn.git && cd sotsuken_nn/gym_myenvs/myenv3/ && mkdir /usr/local/lib/python3.6/dist-packages/gym/envs/myenv3 && cp -rp * /usr/local/lib/python3.6/dist-packages/gym/envs/myenv3 && cd ../ && cp __init__.py /usr/local/lib/python3.6/dist-packages/gym/envs/
RUN echo "successful!"

CMD echo 'Hello docker of sotsuken world!'