FROM python:3.10.17-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Install system packages iproute2, gettext-base plus SSH tools
RUN apt update && apt install -y \
    iproute2 iputils-ping \
    gettext-base \
    openssh-server \
    openssh-client \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    ntplib \
    tensorflow==2.17.1 \
    kafka-python \
    confluent-kafka

# Configure SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:rootpass' | chpasswd  # Set root password for SSH
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config  # Allow password auth
RUN ssh-keygen -A  # Generate SSH host keys

# Copy application code
COPY ./FLInstant /in_network_federaed_learning_for_anomaly_detection
# Set working directory
WORKDIR /in_network_federaed_learning_for_anomaly_detection

# Create a startup script to run SSH server and allow custom commands
RUN echo '#!/bin/bash\n/usr/sbin/sshd\nexec "$@"' > /start.sh
RUN chmod +x /start.sh

# Use the startup script as entrypoint to allow flexibility for server/client
ENTRYPOINT ["/start.sh"]
CMD ["bash"]  # Default to bash; will be overridden for server/client
