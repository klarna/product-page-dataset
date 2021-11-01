FROM python:3.7
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# This is a bit more verbose than you'd expect,
# because it's impossible to refer to variables declared within the same layer
# (e.g. WORK_DIR="/service" PYTHONPATH="$WORK_DIR")
#
# Dependencies are only stored in environment in order not to create needless layers by making them ARGs
ENV WORK_DIR="/treelstm-classifier" \
    VIRTUAL_ENV="/treelstm-classifier/.venv" \
    PATH="/treelstm-classifier/.venv/bin:$PATH" \
    PYTHONPATH="/treelstm-classifier" \
    BUILD_DEPS="apt-utils gnupg2 build-essential unzip xvfb" \
    RUN_DEPS="" \
    USE_XVFB=1 \
    DEBIAN_FRONTEND=noninteractive \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

COPY requirements.txt $WORK_DIR/

# Install prerequisites, do some necessary setup, and clean up
RUN apt-get update && \
    apt-get -y --no-install-recommends install $BUILD_DEPS $RUN_DEPS && \
    python -m venv .venv --prompt "treelstm-classifier" && \
    pip install --no-cache-dir -r $WORK_DIR/requirements.txt && \
    mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc && \
    useradd -m --uid 10000 --shell /bin/bash -m klarnai && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get update && \
    apt-get -y --no-install-recommends install google-chrome-stable npm shellcheck && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Chromedriver 
RUN CHROMEDRIVER_VERSION=$(curl --silent "https://chromedriver.storage.googleapis.com/LATEST_RELEASE") && \
    wget --quiet "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip" -O /chromedriver.zip && \
    mkdir -p /usr/local/bin/ && \
    unzip -o /chromedriver.zip -d /usr/local/bin/ && \
    rm /chromedriver.zip && \
    chmod +x /usr/local/bin/chromedriver


# Copy the code
COPY . $WORK_DIR

# Done
WORKDIR $WORK_DIR
CMD ["/bin/bash"]
