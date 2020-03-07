# always defined on the top to set which image we will be loading from
FROM ubuntu:18.04

# RUN is for running bash commands
RUN apt-get update; \
     apt-get install -y software-properties-common; \
     add-apt-repository ppa:deadsnakes/ppa; \
     apt-get install python3.7 curl -y ; \
     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \
     python3 get-pip.py ; \
     pip3 install torch numpy

RUN pip3 install torch transformers numpy pandas scikit-learn panel==0.7.0 jupyter scipy matplotlib fastprogress

ADD . ./askai

WORKDIR askai/

# CMD is used to set the command which will be run when the container starts
CMD ["panel", "serve", "askai_app.py" ,"--port=5006","--allow-websocket-origin=3.213.106.226", "--allow-websocket-origin=textbookqa.com","--allow-websocket-origin=www.textbookqa.com","--args", "models/2.0/base", "health_education"]
