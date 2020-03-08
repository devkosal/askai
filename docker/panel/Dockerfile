# always defined on the top to set which image we will be loading from
FROM ubuntu:18.04

# RUN is for running bash commands
RUN apt-get update; \
     apt-get install -y software-properties-common; \
     add-apt-repository ppa:deadsnakes/ppa; \
     apt-get install python3.7 curl -y ; \
     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \
     python3 get-pip.py

RUN pip3 install pandas scikit-learn panel==0.7.0 scipy requests

ADD ./src/utils_app.py ./askai/src/utils_app.py
ADD ./askai_app.py ./askai/askai_app.py
ADD ./examples ./askai/examples


WORKDIR askai/

# CMD is used to set the command which will be run when the container starts
CMD ["panel", "serve", "askai_app.py" ,"--port=5006","--allow-websocket-origin=3.213.106.226",\
 "--allow-websocket-origin=textbookqa.com","--allow-websocket-origin=www.textbookqa.com",\
 "--allow-websocket-origin=localhost:5006",\
 "--args", "models/2.0/base", "health_education"]
