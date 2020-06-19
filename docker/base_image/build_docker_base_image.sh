# build docker
DOCKER_PATH="./Dockerfile_base_image_merge"

echo "Trying to build docker image from:" $DOCKER_PATH;

if [ -f "$DOCKER_PATH" ]; then
  echo "file exits"
  sudo docker build -t iainmackie/multi-task-ranking-base-image:v1 -f $DOCKER_PATH .
  # Note already have versions 2,3,4 when trying to merge githubs.

else
  echo "Error - path to file not found:" $DOCKER_PATH;
  exit 0;

fi
