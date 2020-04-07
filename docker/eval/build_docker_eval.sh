# build docker
DOCKER_PATH="./Dockerfile_eval"

echo "Trying to build docker image from:" $DOCKER_PATH;

if [ -f "$DOCKER_PATH" ]; then
  echo "file exits"
  sudo docker build -t iainmackie/multi-task-ranking-eval:v1 -f $DOCKER_PATH .

else
  echo "Error - path to file not found:" $DOCKER_PATH;
  exit 0;

fi
