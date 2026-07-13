FROM jenkins/jenkins:lts-jdk21

USER root

# Install required packages
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    git \
    unzip \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Allow Jenkins user to access Docker
RUN groupadd -f docker && usermod -aG docker jenkins

USER jenkins

# Install Jenkins plugins automatically
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

EXPOSE 8080 50000