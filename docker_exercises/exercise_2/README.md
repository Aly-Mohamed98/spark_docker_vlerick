# Data Minded Academy - Containerization with Docker
## Exercise 2 - Interact with containers
# Changed here
Now that we’ve warmed up it’s time to get inside a container while it’s running! In this exercise, 
you will have to use the `docker` CLI to interact with running containers.

We will use the `devopsdockeruh/simple-web-service:ubuntu` Docker image. Once instantiated as a container,
this image will outputs logs into a file `text.log`. Go inside the container and use `tail -f /usr/src/app/text.log` to 
follow the logs. Every 10 seconds the clock will send you a “secret message”.

1. Run a container from the `devopsdockeruh/simple-web-service:ubuntu` image (in detached mode).

2. Go inside the running container (using the right method) and use `tail -f ./text.log` to follow 
the logs. Every 10 seconds the clock will send you a “secret message”. What is the secret message? 

3. Run another container from the image `devopsdockeruh/simple-web-service:ubuntu` and use the 
second known method to connect inside a running container. You want another method than the one used in (2). 
What is the difference between those two approaches?
