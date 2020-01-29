FROM ubuntu:18.04 # always defined on the top to set which image we will be loading from

# RUN is for running bash commands
RUN apt-get update; \
     apt-get install -y software-properties-common; \
     add-apt-repository ppa:deadsnakes/ppa; \
     apt-get install python3.7 curl -y ; \
     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; \
     python3 get-pip.py ; \
     pip3 install torch numpy

RUN git clone https://github.com/devkosal/albert-qa --depth 1; \
      pip3 install -r "albert-qa/requirements.text"

# CMD is used to set the command which will be run when the container starts
CMD ["python3", "panel", "serve", "--port 5006"]
