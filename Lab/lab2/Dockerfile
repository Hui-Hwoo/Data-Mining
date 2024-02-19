# # Use an official Python runtime as a parent image
# FROM python:3.8

# # Set the working directory in the container
# WORKDIR /app

# # Install Jupyter Lab and its dependencies
# RUN pip install --upgrade pip && \
#     pip install jupyterlab

# # Expose the port that Jupyter Lab will run on
# EXPOSE 8888

# # Start Jupyter Lab when the container launches
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

FROM ubuntu:20.04
RUN apt update && apt install -y sbcl
WORKDIR /usr/src
 