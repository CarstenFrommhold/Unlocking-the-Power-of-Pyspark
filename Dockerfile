FROM python:3.9

WORKDIR tutorial/

COPY ./ ./

RUN apt-get update && apt-get install -y openjdk-11-jdk

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
