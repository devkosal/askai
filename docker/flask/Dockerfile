# always defined on the top to set which image we will be loading from
FROM ubuntu:18.04

# RUN is for running bash commands
RUN apt-get update; \
     apt-get install -y software-properties-common; \
     add-apt-repository ppa:deadsnakes/ppa; \
     apt-get install python3.7 curl -y ; \
     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \
     python3 get-pip.py ; \
     pip3 install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install transformers flask flask_restful gunicorn

ADD ./src/model.py ./askai/src/model.py

ADD ./src/utils_backend.py ./askai/src/utils_backend.py

ADD ./models ./askai/models

ADD ./api.py ./askai/api.py

WORKDIR askai/

# CMD is used to set the command which will be run when the container starts
CMD ["gunicorn", "-w=4", "api:app", "-b=0.0.0.0:5000"]
