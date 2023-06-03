FROM sab148/misato-dataset:latest
USER root
# Set up time zone.

#RUN useradd -m -u 1000 user
#USER user
#ENV HOME=/home/user \
#	PATH=/home/user/.local/bin:$PATH
#WORKDIR $HOME/app
#COPY --chown=user . $HOME/app


#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser .
#
#USER appuser
#
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# Set up a new user named "user" with user ID 1000


RUN mkdir -p /maps
WORKDIR /maps
COPY maps/*pickle .


RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH


WORKDIR $HOME/app


#RUN chmod 777 /data
#RUN useradd -m -u 1000 user
#USER user
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

ENV AMBERHOME="/usr/bin/amber22"
ENV PATH="$AMBERHOME/bin:$PATH"
ENV PYTHONPATH="$AMBERHOME/lib/python3.8/site-packages"

RUN pip install -r requirements.txt
CMD ["python", "main.py"]



