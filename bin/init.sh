
docker-compose -f docker/docker-compose.yml down --remove-orphans

sleep 3

docker build . -t sitemkt -f docker/Dockerfile

docker-compose -f docker/docker-compose.yml up -d

# create tables
docker-compose exec sitemkt-container python bin/main.py init-db



